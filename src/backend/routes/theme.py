from fastapi import APIRouter, HTTPException, Response

from lib.os_theme import render_theme_css

router = APIRouter(prefix="/api", tags=["theme"])


@router.get("/theme.css")
def get_theme_css() -> Response:
    """The user's ~/.os color scheme; 404 without it, so the stylesheet link is a no-op and the bundled fallback stays."""
    css = render_theme_css()
    if css is None:
        raise HTTPException(status_code=404, detail="No ~/.config/theme/palette.json")
    return Response(css, media_type="text/css", headers={"Cache-Control": "no-cache"})
