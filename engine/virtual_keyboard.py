"""
IrisFlow — Teclado Virtual
QWERTY controlado pelo olhar. Seleção por dwell time (1.5 s padrão).
Layout centralizado: 65 % da largura × 55 % da altura do monitor.
"""

import time
from typing import Callable, List, Optional

import cv2
import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Layout e constantes
# ---------------------------------------------------------------------------

KEYBOARD_ROWS: List[List[str]] = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", "APAGAR"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "ENTER"],
    ["ESPACO", "FALAR", "LIMPAR"],
]

_WIDE = {"ESPACO", "FALAR", "LIMPAR"}

_C_KEY   = (55,  55,  55)
_C_HOVER = (90, 130, 200)
_C_FLASH = (30, 180,  80)
_C_TXT   = (240, 240, 240)
_C_PROG  = (0,  210, 255)
_C_DBG   = (18,  18,  18)
_C_DBT   = (255, 255, 255)

_FLASH_DUR = 0.35


def _dwell_for(label: str) -> float:
    """Retorna o dwell time em segundos para o tipo de tecla."""
    if label in {"ESPACO", "FALAR", "LIMPAR"}:
        return 0.8
    if label in {"APAGAR", "ENTER"}:
        return 1.5
    return 1.2


# ---------------------------------------------------------------------------
# Tecla
# ---------------------------------------------------------------------------

class _Key:
    __slots__ = ("label", "x", "y", "w", "h", "_pressed_at")

    def __init__(self, label: str, x: int, y: int, w: int, h: int) -> None:
        self.label       = label
        self.x, self.y   = x, y
        self.w, self.h   = w, h
        self._pressed_at = 0.0

    def hit(self, px: float, py: float) -> bool:
        ex = self.w * 0.10
        ey = self.h * 0.10
        return (self.x - ex) <= px <= (self.x + self.w + ex) and \
               (self.y - ey) <= py <= (self.y + self.h + ey)

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2

    def flash(self) -> None:
        self._pressed_at = time.time()

    @property
    def flashing(self) -> bool:
        return (time.time() - self._pressed_at) < _FLASH_DUR


# ---------------------------------------------------------------------------
# Teclado Virtual
# ---------------------------------------------------------------------------

class VirtualKeyboard:
    """
    Teclado virtual QWERTY centralizado na tela.

    Ocupa 65 % da largura × 55 % da altura do monitor, centrado.
    Teclas maiores facilitam a seleção por dwell time.

    update(cx, cy) — recebe posição do olhar em PIXELS a cada frame.
    draw(canvas)   — desenha teclado + área de texto no canvas.
    """

    _TEXT_H = 88
    _GAP    = 6

    def __init__(
        self,
        screen_width:   int   = 1280,
        screen_height:  int   = 720,
        dwell_time:     float = 1.5,
        on_speak:       Optional[Callable[[str], None]] = None,
        right_reserved: int   = 0,   # mantido para compatibilidade — não utilizado
    ) -> None:
        self.sw          = screen_width
        self.sh          = screen_height
        self.dwell_time  = dwell_time
        self.on_speak    = on_speak
        self.text_buffer = ""

        self._keys:         List[_Key]          = []
        self._hover:        Optional[_Key]      = None
        self._hover_t:      Optional[float]     = None
        self._hover_frames: int                 = 0

        # Geometria do widget (preenchida em _build)
        self._wx = self._wy = 0
        self._ww = screen_width
        self._wh = screen_height
        self._kb_top_y = screen_height - 400

        self._build()
        logger.info(
            "VirtualKeyboard {}×{} dwell={:.1f}s widget={}×{}",
            screen_width, screen_height, dwell_time,
            int(screen_width * 0.65), int(screen_height * 0.55),
        )

    # ------------------------------------------------------------------
    # Propriedade pública
    # ------------------------------------------------------------------

    @property
    def keyboard_top(self) -> int:
        return self._kb_top_y

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self) -> None:
        n_rows = len(KEYBOARD_ROWS)

        # Widget centrado: 65 % largura × 55 % altura
        self._ww = int(self.sw * 0.65)
        self._wh = int(self.sh * 0.55)
        self._wx = (self.sw - self._ww) // 2
        self._wy = (self.sh - self._wh) // 2

        # Área de teclas: abaixo da área de texto
        kb_top    = self._wy + self._TEXT_H + 8
        kb_avail  = self._wh - self._TEXT_H - 8
        gap_total = (n_rows - 1) * self._GAP
        key_h     = max(55, (kb_avail - gap_total) // n_rows)
        self._kb_top_y = kb_top

        self._keys.clear()
        for ri, row in enumerate(KEYBOARD_ROWS):
            y      = kb_top + ri * (key_h + self._GAP)
            n_wide = sum(1 for k in row if k in _WIDE)
            n_norm = len(row) - n_wide
            units  = n_norm + n_wide * 3
            gaps_w = (len(row) - 1) * self._GAP
            unit_w = (self._ww - gaps_w) / max(units, 1)

            x = self._wx
            for label in row:
                w = int(unit_w * (3 if label in _WIDE else 1))
                self._keys.append(_Key(label=label, x=x, y=y, w=w, h=key_h))
                x += w + self._GAP

    # ------------------------------------------------------------------
    # Atualização
    # ------------------------------------------------------------------

    def update(self, gaze_x: float, gaze_y: float) -> None:
        hovered: Optional[_Key] = None
        for key in self._keys:
            if key.hit(gaze_x, gaze_y):
                hovered = key
                break

        if hovered is not self._hover:
            self._hover        = hovered
            self._hover_t      = None
            self._hover_frames = 0
            return

        if hovered is None:
            return

        self._hover_frames += 1
        if self._hover_frames == 3:
            self._hover_t = time.time()

        if self._hover_t is not None:
            dt = _dwell_for(hovered.label)
            if (time.time() - self._hover_t) >= dt:
                self._activate(hovered)
                self._hover_t      = time.time()
                self._hover_frames = 0

    # ------------------------------------------------------------------
    # Ativação
    # ------------------------------------------------------------------

    def _activate(self, key: _Key) -> None:
        key.flash()
        lbl = key.label

        if   lbl == "APAGAR": self.text_buffer = self.text_buffer[:-1]
        elif lbl == "ENTER":  self.text_buffer += "\n"
        elif lbl == "ESPACO": self.text_buffer += " "
        elif lbl == "LIMPAR": self.text_buffer = ""
        elif lbl == "FALAR":
            text = self.text_buffer.strip()
            logger.info("TTS acionado: '{}'", text)
            if text and self.on_speak:
                self.on_speak(text)
            return
        else:
            self.text_buffer += lbl

        logger.debug("Tecla='{}' | buffer='{}'", lbl, self.text_buffer)

    # ------------------------------------------------------------------
    # Desenho
    # ------------------------------------------------------------------

    def draw(self, canvas: np.ndarray) -> np.ndarray:
        self._draw_text_area(canvas)
        self._draw_keys(canvas)
        return canvas

    def _draw_text_area(self, canvas: np.ndarray) -> None:
        x0, y0 = self._wx, self._wy
        x1, y1 = self._wx + self._ww, self._wy + self._TEXT_H

        cv2.rectangle(canvas, (x0, y0), (x1, y1), _C_DBG, -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (90, 90, 90), 2)
        cv2.putText(canvas, "Texto:", (x0 + 14, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 1)

        text      = self.text_buffer.replace("\n", " | ")
        cursor    = "|" if int(time.time() * 2) % 2 == 0 else ""
        full      = text + cursor
        max_chars = max(1, (x1 - x0 - 30) // 15)
        tail      = full[-max_chars:] if len(full) > max_chars else full
        if not text:
            tail = cursor or "_"

        cv2.putText(canvas, tail, (x0 + 18, y1 - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.90, _C_DBT, 2)

    def _draw_keys(self, canvas: np.ndarray) -> None:
        now = time.time()
        for key in self._keys:
            is_hover = key is self._hover
            is_flash = key.flashing

            bg     = _C_FLASH if is_flash else (_C_HOVER if is_hover else _C_KEY)
            border = (160, 160, 160) if is_hover else (75, 75, 75)

            cv2.rectangle(canvas, (key.x, key.y), (key.x + key.w, key.y + key.h), bg, -1)
            cv2.rectangle(canvas, (key.x, key.y), (key.x + key.w, key.y + key.h), border, 1)

            if is_hover and not is_flash and self._hover_t is not None:
                dt    = _dwell_for(key.label)
                prog  = min(1.0, (now - self._hover_t) / dt)
                bar_w = int(key.w * prog)
                if bar_w > 0:
                    bar_color = (
                        0,
                        int(210 + 10 * prog),
                        int(255 - 175 * prog),
                    )
                    cv2.rectangle(
                        canvas,
                        (key.x,         key.y + key.h - 8),
                        (key.x + bar_w, key.y + key.h),
                        bar_color, -1,
                    )

            lbl = key.label
            fs  = (0.80 if len(lbl) == 1 else
                   0.72 if len(lbl) <= 3 else
                   0.65 if lbl in _WIDE else 0.52)

            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            cv2.putText(canvas, lbl,
                        (key.cx - tw // 2, key.cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, _C_TXT, 2)
