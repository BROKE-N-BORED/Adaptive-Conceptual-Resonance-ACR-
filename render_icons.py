#!/usr/bin/env python3
"""Render icons/acr-48.png and icons/acr-256.png from icons/acr.svg.

Build-time helper, shipped in the zip: build_zip.py calls it when
the PNGs are missing, as in a fresh clone (the repo holds only the
SVG). Uses QSvgRenderer when available, else draws the same shapes
with QPainter. Verifies the PNGs by reading them back.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def render_with_svg(app):
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtCore import QSize
    renderer = QSvgRenderer(os.path.join(ROOT, "icons", "acr.svg"))
    if not renderer.isValid():
        return False
    for size in (48, 256):
        image = QImage(size, size, QImage.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        out = os.path.join(ROOT, "icons", "acr-%d.png" % size)
        image.save(out, "PNG")
    return True


def render_with_painter(app):
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
    for size in (48, 256):
        scale = size / 256.0
        image = QImage(size, size, QImage.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.scale(scale, scale)
        painter.setBrush(QColor("#14202e"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(QRectF(8, 8, 240, 240), 48, 48)
        painter.setBrush(QColor("#f2c14e"))
        painter.drawEllipse(QRectF(112, 112, 32, 32))
        pen = QPen(QColor("#7fd4ff"))
        pen.setWidth(10)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawArc(QRectF(88, 88, 80, 80), 0, 90 * 16)
        painter.drawArc(QRectF(56, 56, 144, 144), 0, 90 * 16)
        painter.drawArc(QRectF(24, 24, 208, 208), 0, 90 * 16)
        pen.setColor(QColor("#f2c14e"))
        painter.setPen(pen)
        painter.drawArc(QRectF(88, 88, 80, 80), 180 * 16, 90 * 16)
        painter.drawArc(QRectF(56, 56, 144, 144), 180 * 16, 90 * 16)
        painter.end()
        out = os.path.join(ROOT, "icons", "acr-%d.png" % size)
        image.save(out, "PNG")
    return True


def main():
    from PySide6.QtGui import QGuiApplication, QImage
    app = QGuiApplication(sys.argv)
    try:
        used = "svg" if render_with_svg(app) else "painter"
    except ImportError:
        render_with_painter(app)
        used = "painter"
    ok = True
    for size in (48, 256):
        path = os.path.join(ROOT, "icons", "acr-%d.png" % size)
        image = QImage(path)
        good = (not image.isNull() and image.width() == size
                and image.height() == size
                and os.path.getsize(path) > 100)
        print("%s: %dx%d, %d bytes, ok=%s"
              % (path, image.width(), image.height(),
                 os.path.getsize(path), good))
        ok = ok and good
    print("renderer: %s" % used)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
