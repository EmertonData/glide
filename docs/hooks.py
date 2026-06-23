import logging
import os
import re

log = logging.getLogger("mkdocs")


def on_page_content(html, page, config, **kwargs):
    for src in re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE):
        if src.startswith(("http://", "https://", "data:")):
            continue
        page_dir = os.path.dirname(page.file.src_path)
        img_path = os.path.normpath(os.path.join(config["docs_dir"], page_dir, src))
        if not os.path.exists(img_path):
            log.warning("Image not found: '%s' in %s", src, page.file.src_path)
