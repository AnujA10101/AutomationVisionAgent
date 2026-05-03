"""Application entrypoint: wires UI, overlay, agent, and background work."""

from __future__ import annotations

import logging
import sys

from PyQt6.QtCore import QEvent, QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication

from app.agent import AutomationAgent
from app.config import SETTINGS
from app.controller import AgentController
from app.executor import ActionExecutor
from app.llm import OpenAIClient
from app.logging_config import configure_logging
from app.overlay import OverlayWindow
from app.ui import SidePanel
from app.vision import ScreenCapture

logger = logging.getLogger(__name__)


class _EscapeStopFilter(QObject):
    """Global Escape → emergency stop (shared controller)."""

    def __init__(self, stop_fn: object) -> None:
        super().__init__()
        self._stop_fn = stop_fn

    def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:  # noqa: ARG002
        if event is not None and event.type() == QEvent.Type.KeyPress:
            key_event = event
            if isinstance(key_event, QKeyEvent) and key_event.key() == Qt.Key.Key_Escape:
                self._stop_fn()  # type: ignore[misc]
        return False


class _AgentWorker(QObject):
    """Runs `AutomationAgent.run_task` off the UI thread."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(self, agent: AutomationAgent) -> None:
        super().__init__()
        self._agent = agent
        self._prompt = ""

    def configure(self, prompt: str) -> None:
        self._prompt = prompt

    def run(self) -> None:
        try:
            self._agent.run_task(
                self._prompt,
                on_status=self.progress.emit,
            )
            self.finished.emit(True, "")
        except Exception as exc:  # noqa: BLE001 — top-level worker boundary
            logger.exception("Agent run failed")
            self.finished.emit(False, str(exc))


class MainController(QObject):
    """Owns long-lived objects and bridges worker signals to UI/overlay."""

    def __init__(
        self,
        qt_app: QApplication,
        panel: SidePanel,
        overlay: OverlayWindow,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._panel = panel
        self._overlay = overlay

        self._thread: QThread | None = None
        self._worker: _AgentWorker | None = None

        self._controller = AgentController()
        panel.set_stop_handler(self._controller.request_stop)

        executor = ActionExecutor(overlay=overlay, controller=self._controller)
        capture = ScreenCapture()
        llm = OpenAIClient()
        self._agent = AutomationAgent(
            llm=llm,
            executor=executor,
            capture=capture,
            controller=self._controller,
            overlay=overlay,
        )

        self._panel.prompt_submitted.connect(self._on_prompt_submitted)

        self._escape_filter = _EscapeStopFilter(self._controller.request_stop)
        qt_app.installEventFilter(self._escape_filter)

    @pyqtSlot(str)
    def _on_prompt_submitted(self, prompt: str) -> None:
        if self._thread is not None:
            self._panel.set_status("Busy: please wait for the current run to finish.")
            return

        self._panel.set_running(True)
        self._panel.set_status("Starting…")

        thread = QThread()
        worker = _AgentWorker(self._agent)
        worker.configure(prompt)
        worker.moveToThread(thread)

        worker.progress.connect(self._panel.set_status)
        worker.finished.connect(self._on_worker_finished)

        thread.started.connect(worker.run)

        self._thread = thread
        self._worker = worker
        thread.start()

    @pyqtSlot(bool, str)
    def _on_worker_finished(self, ok: bool, err: str) -> None:
        self._panel.set_running(False)
        if not ok:
            self._panel.set_status(f"Error: {err}")
            logger.error("Run finished with error: %s", err)

        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()

        self._thread = None
        self._worker = None


def main() -> int:
    """Start the Qt event loop and show the control panel + overlay."""
    configure_logging()

    if SETTINGS.app.debug:
        print(f"[main] Starting {SETTINGS.app.name} (debug={SETTINGS.app.debug})")

    qt_app = QApplication(sys.argv)

    overlay = OverlayWindow()
    overlay.show()

    panel = SidePanel()
    panel.show()

    _ = MainController(qt_app, panel, overlay, parent=panel)

    return qt_app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
