#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Dict, List

from log_saver_node import save_sequential_data


@dataclass
class DataLogger:
    """GUI와 저장 로직을 분리하기 위한 기록 관리 클래스."""
    record_buffer: List[Dict[str, object]] = field(default_factory=list)
    is_recording: bool = False

    def start(self) -> None:
        self.is_recording = True
        self.record_buffer = []

    def stop_and_save(self) -> Dict[str, object]:
        self.is_recording = False
        result = save_sequential_data(self.record_buffer)
        if result.get("status") == "saved":
            self.record_buffer = []
        return result

    def add_record(self, data_row: Dict[str, object]) -> None:
        if self.is_recording:
            self.record_buffer.append(data_row)

    @property
    def record_count(self) -> int:
        return len(self.record_buffer)
