"""
image_generator.py — Christian image generation via pollinations.ai.
Free, no API key, no rate limits for demo use.
Enforces Christian-appropriate style via prompt engineering.
"""

import urllib.parse
import httpx
from safety.moderator import moderate_image_prompt

POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"

# Style suffix appended to all prompts to enforce Christian aesthetic
STYLE_SUFFIX = (
    ", biblical illustration, sacred christian art, reverent atmosphere, "
    "warm divine light, high detail, masterpiece, spiritual, inspirational"
)

# Negative concepts encoded in prompt (pollinations supports natural language)
NEGATIVE_PREFIX = "A reverent Christian artwork (no violence, no explicit content, no dark imagery): "


def generate_christian_image(user_prompt: str) -> dict:
    """
    Validate prompt → enhance with style → pre-fetch from pollinations.ai → return URL.

    The backend waits for pollinations.ai to generate the image (up to 120s)
    before returning, so the frontend img tag loads immediately.

    Returns:
        {
          "success": bool,
          "url": str | None,
          "blocked": bool,
          "reason": str | None,
          "enhanced_prompt": str | None
        }
    """
    # Moderation check
    mod = moderate_image_prompt(user_prompt)
    if not mod.allowed:
        return {
            "success": False,
            "url": None,
            "blocked": True,
            "reason": mod.reason,
            "enhanced_prompt": None,
        }

    # Enhance prompt
    enhanced = NEGATIVE_PREFIX + user_prompt + STYLE_SUFFIX

    # Build URL (pollinations uses URL-encoded prompt in path)
    encoded = urllib.parse.quote(enhanced)
    image_url = f"{POLLINATIONS_BASE}/{encoded}?width=768&height=512&seed=42&model=flux"

    # Pre-fetch: wait until pollinations.ai has finished generating the image.
    # seed=42 means the same prompt always returns the same cached image on retry.
    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            resp = client.get(image_url)
            resp.raise_for_status()
    except Exception as exc:
        return {
            "success": False,
            "url": None,
            "blocked": False,
            "reason": f"Image generation timed out or failed: {exc}",
            "enhanced_prompt": enhanced,
        }

    return {
        "success": True,
        "url": image_url,
        "blocked": False,
        "reason": None,
        "enhanced_prompt": enhanced,
    }
