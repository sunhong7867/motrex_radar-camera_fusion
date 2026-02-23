#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GUI 기록 버퍼 관리 유틸리티
- 기록 시작/종료 상태 관리
- 버퍼 누적 데이터 저장 연동
"""

from dataclasses import dataclass, field
from typing import Dict, List
from log_saver_node import save_sequential_data

@dataclass
# 기록 상태와 버퍼를 관리하는 데이터 로거
class DataLogger:
    record_buffer: List[Dict[str, object]] = field(default_factory=list)
    is_recording: bool = False

    # 기록 시작 시 버퍼 초기화
    def start(self) -> None:
        self.is_recording = True
        self.record_buffer = []

    # 기록 종료 후 누적 버퍼 저장
    def stop_and_save(self) -> Dict[str, object]:
        self.is_recording = False
        # 저장 성공 시 메모리 버퍼 정리
        result = save_sequential_data(self.record_buffer)
        if result.get("status") == "saved":
            self.record_buffer = []
        return result

    # 기록 활성 상태일 때만 단일 행 추가
    def add_record(self, data_row: Dict[str, object]) -> None:
        if self.is_recording:
            self.record_buffer.append(data_row)

    @property
    # 현재 버퍼 길이 반환
    def record_count(self) -> int:
        return len(self.record_buffer)