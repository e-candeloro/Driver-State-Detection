import os
import sys
from pathlib import Path

SYSTEM_QT_FONT_DIRS = (
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/TTF"),
    Path("/usr/share/fonts"),
)


def configure_qt_before_cv2_import():
    """Select OpenCV wheel's XCB plugin and isolate it from KDE 6 theme state."""
    if not sys.platform.startswith("linux"):
        return

    using_xwayland = (
        os.environ.get("XDG_SESSION_TYPE") == "wayland"
        and bool(os.environ.get("DISPLAY"))
        and "QT_QPA_PLATFORM" not in os.environ
    )
    if using_xwayland:
        # PyPI's OpenCV Qt wheel ships libqxcb but currently no Wayland plugin.
        os.environ["QT_QPA_PLATFORM"] = "xcb"

    if (
        os.environ.get("QT_QPA_PLATFORM") == "xcb"
        and os.environ.get("KDE_SESSION_VERSION") == "6"
    ):
        # Bundled Qt 5 cannot parse KDE 6's serialized font descriptions.
        os.environ["XDG_CURRENT_DESKTOP"] = ""
        os.environ["KDE_FULL_SESSION"] = ""
        os.environ["DESKTOP_SESSION"] = ""


def configure_qt_fonts_after_cv2_import():
    """Replace the wheel's missing Qt font directory with an installed system path."""
    if not sys.platform.startswith("linux"):
        return
    configured = Path(os.environ.get("QT_QPA_FONTDIR", ""))
    if configured.is_dir():
        return
    for font_dir in SYSTEM_QT_FONT_DIRS:
        if font_dir.is_dir():
            os.environ["QT_QPA_FONTDIR"] = str(font_dir)
            return
