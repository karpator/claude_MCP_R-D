from typing import Dict, List, Union, Any, Optional
from pocketflow_models import MessageParts, FilePart, ImagePart, TextPart, ContentPart
import re


def parse_history(history: Optional[List[Union[MessageParts, Dict[str, Any]]]],
                  message: Optional[Union[MessageParts, Dict[str, Any]]]) -> \
        List[MessageParts]:
    parsed_his = [MessageParts.model_validate(h) if isinstance(h, dict) else h for h in
                  history] if history else []
    parsed_msg = [MessageParts.model_validate(message)] if isinstance(message, dict) else [
        message] if message else []
    return parsed_his + parsed_msg


def parse_jinja_template(parsed_jinja: str):
    patterns = {
        'file': (r'!\[\s*file\s*\?\s*(.*)]\((\s*[^{}\n]+\s*)\)',
                 lambda mime, url: FilePart(url=url, content_type=mime)),
        'image': (r'!\[\s*image\s*\?\s*(.*)]\((\s*[^{}\n]+\s*)\)',
                  lambda mime, data: ImagePart(url=data, content_type=mime)),
    }
    lines = parsed_jinja.split('\n')
    parts: List[ContentPart] = []
    buffer = ""
    for line in lines:
        matched = False
        for pattern, constructor in patterns.values():
            match = re.match(pattern, line)
            if match:
                if buffer.strip():
                    parts.append(TextPart(text=buffer.strip()))
                    buffer = ""
                mime, data = match.groups()
                parts.append(constructor(mime.strip(), data.strip()))
                matched = True
                break
        if not matched:
            buffer += line + "\n"

    if buffer.strip():
        parts.append(TextPart(text=buffer.strip()))
    return parts


if __name__ == "__main__":
    test_str = """This is a test message.
![image?image/png](gs://example-bucket/image.png)
Here is a file:
![file?application/pdf](gs://example-bucket/document.pdf)
End of message."""
    result = parse_jinja_template(test_str)
    for part in result:
        print(part)
