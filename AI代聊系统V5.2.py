from __future__ import annotations

"""
AI代聊系统V5.2

兼容性与部署稳定收敛版。

设计目标：
1. 以 V5.1 继续优化版为唯一功能基线，不删减任何已有能力。
2. 通过模块加载方式，避免原始大文件里多次追加定义与过早 __main__ 调用带来的
   启动版本不确定问题。
3. 在不改动原始基线文件的前提下，提供一个单一、明确、可直接部署的入口文件。
4. 对已知的同步发送 fallback 路径做轻量补丁，避免 delay_seconds 导致主线程阻塞。
"""

import importlib.util
import pathlib
import sys
from types import ModuleType
from typing import Any

BASE_FILENAME = "智能聊天_中英分离_深度升级V5.1_继续优化版.py"
BASE_MODULE_NAME = "_tg_ai_chat_v51_continued_base"


def _load_base_module() -> ModuleType:
    base_path = pathlib.Path(__file__).with_name(BASE_FILENAME)
    if not base_path.exists():
        raise FileNotFoundError(
            f"未找到 V5.1 基线文件: {base_path}. "
            f"请确保 {BASE_FILENAME} 与当前文件位于同一目录。"
        )

    spec = importlib.util.spec_from_file_location(BASE_MODULE_NAME, base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法为基线文件创建导入规格: {base_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[BASE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _patch_sender_service(module: ModuleType) -> None:
    """
    兼容性补丁：
    原始连续升级文件中，SenderService 在没有 outbound_queue 时会走同步 sleep fallback。
    这里保留全部功能，但去掉 sleep，避免主路径阻塞。
    """
    sender_service_cls = getattr(module, "SenderService", None)
    if sender_service_cls is None:
        return

    original = getattr(sender_service_cls, "send_text_reply", None)
    if original is None:
        return

    def send_text_reply(
        self,
        conversation_id: int,
        text: str,
        delay_seconds: int = 0,
        raw_payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        outbound_queue = getattr(self, "outbound_queue", None)
        if outbound_queue is not None:
            job_id = outbound_queue.enqueue_text_reply(
                conversation_id=conversation_id,
                text=text,
                raw_payload=raw_payload,
                delay_seconds=delay_seconds,
                metadata=metadata,
            )
            return {"queued": True, "job_id": job_id}

        # 兼容旧路径：保留直接发送能力，但不再阻塞 sleep。
        self.sender_adapter.send_text(conversation_id, text)
        return {"queued": False, "job_id": None, "delay_skipped": delay_seconds > 0}

    setattr(sender_service_cls, "send_text_reply", send_text_reply)


def _export_public_names(module: ModuleType) -> None:
    current_globals = globals()
    for name, value in module.__dict__.items():
        if name.startswith("__"):
            continue
        if name in {
            "importlib",
            "pathlib",
            "sys",
            "ModuleType",
            "Any",
        }:
            continue
        current_globals[name] = value


_BASE_MODULE = _load_base_module()
_patch_sender_service(_BASE_MODULE)
_export_public_names(_BASE_MODULE)

# 固定最终部署入口与组件装配入口。
build_app_components = getattr(_BASE_MODULE, "build_app_components")
create_web_app = getattr(_BASE_MODULE, "create_web_app", None)
main = getattr(_BASE_MODULE, "main")


if __name__ == "__main__":
    result = main()
    raise SystemExit(0 if result is None else result)
