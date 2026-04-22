"""
IrisFlow — Teclado Virtual
Teclado QWERTY controlado pelo olhar. O usuário seleciona uma tecla
fixando o olhar por um tempo determinado (dwell time).
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
from loguru import logger


KEYBOARD_LAYOUT = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", "⌫"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "↵"],
    ["ESPAÇO", "FALAR"],
]

# Cores (BGR)
COLOR_KEY_BG = (50, 50, 50)
COLOR_KEY_HOVER = (80, 120, 200)
COLOR_KEY_ACTIVE = (0, 180, 100)
COLOR_TEXT = (255, 255, 255)
COLOR_PROGRESS = (0, 200, 255)


@dataclass
class Key:
    """Representa uma tecla do teclado virtual."""
    label: str
    x: int
    y: int
    width: int
    height: int

    def contains(self, px: float, py: float, frame_w: int, frame_h: int) -> bool:
        """Verifica se a posição (normalizada) está sobre esta tecla."""
        abs_x = int(px * frame_w)
        abs_y = int(py * frame_h)
        return self.x <= abs_x <= self.x + self.width and self.y <= abs_y <= self.y + self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


class VirtualKeyboard:
    """
    Teclado virtual controlado pelo olhar.

    O usuário seleciona uma tecla fixando o olhar nela por `dwell_time` segundos.
    """

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        dwell_time: float = 1.5,
        on_key_press: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.dwell_time = dwell_time
        self.on_key_press = on_key_press

        self.text_buffer: str = ""
        self._keys: List[Key] = []
        self._hovered_key: Optional[Key] = None
        self._hover_start: float = 0.0

        self._build_layout()
        logger.info("Teclado virtual inicializado (dwell_time={}s)", dwell_time)

    def _build_layout(self) -> None:
        """Constrói a grade de teclas baseada nas dimensões do frame."""
        key_h = 60
        key_margin = 4
        keyboard_top = self.frame_height - (len(KEYBOARD_LAYOUT) * (key_h + key_margin)) - 20

        for row_i, row in enumerate(KEYBOARD_LAYOUT):
            y = keyboard_top + row_i * (key_h + key_margin)
            # Teclas especiais têm largura maior
            normal_keys = [k for k in row if k not in ("ESPAÇO", "FALAR")]
            special_keys = [k for k in row if k in ("ESPAÇO", "FALAR")]

            cols = len(normal_keys) + len(special_keys) * 2  # Especiais = 2x largura
            key_w = (self.frame_width - 2 * 20 - (len(row) - 1) * key_margin) // cols

            x = 20
            for key_label in row:
                w = key_w * 2 if key_label in ("ESPAÇO", "FALAR") else key_w
                self._keys.append(Key(label=key_label, x=x, y=y, width=w, height=key_h))
                x += w + key_margin

    def update(self, gaze_x: float, gaze_y: float) -> None:
        """
        Atualiza o estado do teclado com a posição do olhar.

        Args:
            gaze_x: Posição X normalizada (0.0–1.0)
            gaze_y: Posição Y normalizada (0.0–1.0)
        """
        hovered = None
        for key in self._keys:
            if key.contains(gaze_x, gaze_y, self.frame_width, self.frame_height):
                hovered = key
                break

        if hovered != self._hovered_key:
            self._hovered_key = hovered
            self._hover_start = time.time()
            return

        if hovered and (time.time() - self._hover_start) >= self.dwell_time:
            self._press_key(hovered)
            self._hover_start = time.time()  # Evita pressionar múltiplas vezes

    def _press_key(self, key: Key) -> None:
        """Processa o pressionamento de uma tecla."""
        label = key.label
        if label == "⌫":
            self.text_buffer = self.text_buffer[:-1]
        elif label == "↵":
            self.text_buffer += "\n"
        elif label == "ESPAÇO":
            self.text_buffer += " "
        elif label == "FALAR":
            logger.info("TTS: '{}'", self.text_buffer)
            if self.on_key_press:
                self.on_key_press(f"__SPEAK__:{self.text_buffer}")
            self.text_buffer = ""
            return
        else:
            self.text_buffer += label

        logger.debug("Tecla pressionada: '{}' | Buffer: '{}'", label, self.text_buffer)
        if self.on_key_press:
            self.on_key_press(label)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Desenha o teclado no frame."""
        now = time.time()

        for key in self._keys:
            is_hovered = key == self._hovered_key
            color = COLOR_KEY_HOVER if is_hovered else COLOR_KEY_BG

            # Fundo da tecla
            cv2.rectangle(frame, (key.x, key.y), (key.x + key.width, key.y + key.height), color, -1)
            cv2.rectangle(frame, (key.x, key.y), (key.x + key.width, key.y + key.height), (100, 100, 100), 1)

            # Barra de progresso (dwell)
            if is_hovered:
                progress = min(1.0, (now - self._hover_start) / self.dwell_time)
                bar_w = int(key.width * progress)
                cv2.rectangle(
                    frame,
                    (key.x, key.y + key.height - 4),
                    (key.x + bar_w, key.y + key.height),
                    COLOR_PROGRESS,
                    -1,
                )

            # Texto da tecla
            font_scale = 0.5 if len(key.label) > 2 else 0.7
            text_size = cv2.getTextSize(key.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            tx = key.center[0] - text_size[0] // 2
            ty = key.center[1] + text_size[1] // 2
            cv2.putText(frame, key.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, 1)

        # Buffer de texto
        cv2.rectangle(frame, (20, 10), (self.frame_width - 20, 70), (30, 30, 30), -1)
        cv2.putText(
            frame,
            self.text_buffer[-80:] or "_",  # Mostra últimos 80 chars
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            COLOR_TEXT,
            2,
        )
        return frame
