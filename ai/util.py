def strip_think_block(text: str, think_open: bool) -> tuple[str, bool]:
    offset = 0
    result = ""
    while True:
        # Find next <think> or </think> tag
        tag = "</think>" if think_open else "<think>"
        next_offset = text.find(tag, offset)

        # ... or end of string
        if next_offset == -1:
            next_offset = len(text)

        # Append any non-think content
        if not think_open:
            result += text[offset:next_offset]
        
        # Exit if found end of string
        if next_offset == len(text):
            return result, think_open

        # Toggle think open and skip tag
        think_open = not think_open
        offset = next_offset + len(tag)
