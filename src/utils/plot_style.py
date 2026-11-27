"""포트폴리오 및 블로그 게재용 시각화 스타일·한글 폰트·고해상도 저장 유틸리티."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from src.utils.config import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

# 세그먼트 A~E 전용 시맨틱 팔레트 (소수 VIP A, B 강조 + C, D, E 단계적 대비)
SEGMENT_ORDER: list[str] = ["A", "B", "C", "D", "E"]
SEGMENT_PALETTE: dict[str, str] = {
    "A": "#D62728",  # Crimson Red (최상위 초소수 0.041%)
    "B": "#FF7F0E",  # Vivid Amber (극소수 0.006%, 24명)
    "C": "#2CA02C",  # Emerald Green (상위 5.32%)
    "D": "#1F77B4",  # Royal Blue (중상위 14.55%)
    "E": "#6C757D",  # Slate Gray (다수 일반군 80.09%)
}


def apply_portfolio_style() -> str:
    """한글 폰트 깨짐 방지 및 깔끔한 출판급 차트 테마 설정."""
    sns.set_theme(style="whitegrid", context="notebook")

    candidates = ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen_font = next((c for c in candidates if c in available), "sans-serif")

    plt.rcParams.update(
        {
            "font.family": chosen_font,
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.edgecolor": "#CCCCCC",
            "grid.alpha": 0.35,
        }
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return chosen_font


def save_portfolio_figure(fig: plt.Figure, filename: str) -> Path:
    """차트를 notebooks/figures/ 아래에 300 DPI PNG로 저장."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not filename.endswith(".png"):
        filename = f"{filename}.png"
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    return out_path
