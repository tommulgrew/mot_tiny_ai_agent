from enum import Enum
from typing import Callable, Iterable
from pydantic import BaseModel
from ai.message_accessor import AIMessageAccessor
from ai.message_history import AIChatMessageGroup, AIChatMessageHistory
from ai.util import strip_think_block

class TrimTokensOpType(int, Enum):
    STRIP_THINK = 0
    REMOVE_SYS_INFO = 1
    TRUNCATE_TOOL_RESULTS = 2
    REMOVE_GROUP = 3

TRIM_TOKENS_OP_WEIGHTS = [0.1, 0.25, 0.4, 1.0]
TRIM_TOKENS_AGE_WEIGHTS = [1.0, 0.6, 0.4, 0.3, 0.2, 0.1]

class TrimTokensOp(BaseModel):
    group: AIChatMessageGroup
    age: int                    # 0 = newest
    op_type: TrimTokensOpType

    def get_weight(self) -> float:
        op_weight = TRIM_TOKENS_OP_WEIGHTS[self.op_type]
        age_weight = (
            TRIM_TOKENS_AGE_WEIGHTS[self.age] 
            if self.age < len(TRIM_TOKENS_AGE_WEIGHTS) 
            else TRIM_TOKENS_AGE_WEIGHTS[-1]
        )
        return op_weight * age_weight

TOOL_CONTENT_TRUNCATE_LENGTH = 100

class AIContextManager:
    """Manages an AI chat message history, performing various compactions to keep it within a specified limit"""

    def __init__(self, message_accessor: AIMessageAccessor):
        self.message_accessor = message_accessor

    def trim_to_limit(self, history: AIChatMessageHistory, token_limit: int, is_system_info_callback: Callable[[str], bool] | None) -> bool:
        while history.token_count > token_limit:
            if not self.trim(history, is_system_info_callback):
                return False
        return True

    def trim(self, history: AIChatMessageHistory, is_system_info_callback: Callable[[str], bool] | None) -> bool:

        # Split messages into groups. Exclude last group, as we cannot trim tokens
        # from the current active group.
        groups = history.get_groups()[:-1]
        if not groups:
            return False

        # Create token trim operations for each group
        ops = [ 
            TrimTokensOp(
                group=g, 
                age=len(groups) - idx - 1, 
                op_type=self._get_trim_tokens_op_type(history.group_messages(g), is_system_info_callback)
            ) 
            for idx, g in enumerate(groups)
        ]

        # Sort by weight
        ops.sort(key=lambda x: x.get_weight())

        # Apply lowest weighted operation
        self._do_trim_op(ops[0], history, is_system_info_callback)
        return True

    def _get_trim_tokens_op_type(self, group_messages: list, is_system_info_callback: Callable[[str], bool] | None) -> TrimTokensOpType:
        if self._group_has_think_block(group_messages):
            return TrimTokensOpType.STRIP_THINK
        
        elif self._group_has_sys_info(group_messages, is_system_info_callback):
            return TrimTokensOpType.REMOVE_SYS_INFO

        elif self._group_has_untruncated_tools(group_messages):
            return TrimTokensOpType.TRUNCATE_TOOL_RESULTS

        else:
            return TrimTokensOpType.REMOVE_GROUP

    def _do_trim_op(self, op: TrimTokensOp, history: AIChatMessageHistory, is_system_info_callback: Callable[[str], bool] | None):

        if (op.op_type == TrimTokensOpType.STRIP_THINK):
            self._strip_think_blocks(history, op.group)

        elif (op.op_type == TrimTokensOpType.REMOVE_SYS_INFO):
            self._remove_sys_info(history, op.group, is_system_info_callback)

        elif (op.op_type == TrimTokensOpType.TRUNCATE_TOOL_RESULTS):
            self._truncate_tool_results(history, op.group)

        elif (op.op_type == TrimTokensOpType.REMOVE_GROUP):
            self._remove_group(history, op.group)

    def _group_has_think_block(self, messages: list) -> bool:
        return any(
            self.message_accessor.is_assistant_message(m) and "<think>" in self.message_accessor.get_content(m) 
            for m in messages
        )

    def _group_has_sys_info(self, messages: list, is_system_info_callback: Callable[[str], bool] | None) -> bool:
        if not is_system_info_callback:
            return False
        return any(
            self.message_accessor.is_user_message(m) and is_system_info_callback(self.message_accessor.get_content(m))
            for m in messages
        )

    def _group_has_untruncated_tools(self, messages: list) -> bool:
        return any(
            self.message_accessor.is_tool_message(m) and len(self.message_accessor.get_content(m)) > TOOL_CONTENT_TRUNCATE_LENGTH
            for m in messages
        )

    def _strip_think_blocks(self, history: AIChatMessageHistory, group: AIChatMessageGroup):
        group_messages = history.group_messages(group)
        think_open = False
        for i, msg in enumerate(group_messages):
            if self.message_accessor.is_assistant_message(msg):
                content = self.message_accessor.get_content(msg)
                content, think_open = strip_think_block(content, think_open)
                history.set_content(group.start_index + i, content)

    def _remove_sys_info(self, history: AIChatMessageHistory, group: AIChatMessageGroup, is_system_info_callback: Callable[[str], bool] | None):
        if not is_system_info_callback:
            return
        group_messages = history.group_messages(group)
        history.remove(
            msg
            for msg in group_messages 
            if self.message_accessor.is_user_message(msg) and is_system_info_callback(self.message_accessor.get_content(msg))
        )

    def _truncate_tool_results(self, history: AIChatMessageHistory, group: AIChatMessageGroup):
        group_messages = history.group_messages(group)
        for i, msg in enumerate(group_messages):
            if self.message_accessor.is_tool_message(msg):
                content = self.message_accessor.get_content(msg)
                if len(content) > TOOL_CONTENT_TRUNCATE_LENGTH:
                    new_content = content[:97] + "..."
                    history.set_content(group.start_index + i, new_content)

    def _remove_group(self, history: AIChatMessageHistory, group: AIChatMessageGroup):
        group_messages = history.group_messages(group)
        history.remove(group_messages)
