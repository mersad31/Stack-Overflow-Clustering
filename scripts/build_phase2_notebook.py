from __future__ import annotations

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}}
    nb.cells = [
        nbf.v4.new_markdown_cell("# فاز دوم — مقایسه و اعتبارسنجی خوشه‌بندی\n\nتمام اعداد مستقیما از artifactهای pipeline خوانده می‌شوند."),
        nbf.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display, Image\nROOT=Path.cwd().resolve()\nif ROOT.name=='notebooks': ROOT=ROOT.parent\nsummary=json.loads((ROOT/'artifacts/phase2_run_summary.json').read_text(encoding='utf-8'))\nsummary['elapsed_seconds']"),
        nbf.v4.new_markdown_cell("## انتخاب k و الگوریتم‌های partitioning"),
        nbf.v4.new_code_cell("pd.read_csv(ROOT/'reports/tables/phase2_k_selection_synthesis.csv').sort_values('selection_rank_sum')[['k','silhouette','davies_bouldin','calinski_harabasz','mean','selection_rank_sum']]"),
        nbf.v4.new_code_cell("display(Image(filename=ROOT/'reports/figures/phase2_partitioning_selection.png', width=900))"),
        nbf.v4.new_code_cell("display(Image(filename=ROOT/'reports/figures/phase2_bootstrap_k_stability.png', width=800))"),
        nbf.v4.new_markdown_cell("## مقایسه خانواده‌های مدل"),
        nbf.v4.new_code_cell("scores=pd.read_csv(ROOT/'reports/tables/phase2_model_family_scores.csv')\nscores.query('minimum_cluster_fraction >= 0.02 and silhouette == silhouette').sort_values('silhouette',ascending=False).head(15)[['family','algorithm','k_requested','silhouette','davies_bouldin','minimum_cluster_fraction','noise_fraction']]"),
        nbf.v4.new_code_cell("display(Image(filename=ROOT/'reports/figures/phase2_family_silhouette_comparison.png', width=900))"),
        nbf.v4.new_markdown_cell("## GMM، روش‌های چگالی و سلسله‌مراتبی"),
        nbf.v4.new_code_cell("display(Image(filename=ROOT/'reports/figures/phase2_gmm_information_criteria.png', width=900))"),
        nbf.v4.new_code_cell("display(Image(filename=ROOT/'reports/figures/phase2_hierarchical_dendrograms.png', width=950))"),
        nbf.v4.new_code_cell("display(Image(filename=ROOT/'reports/figures/phase2_density_tradeoff.png', width=800))"),
        nbf.v4.new_markdown_cell("## توافق، پایداری و محدودیت نتیجه"),
        nbf.v4.new_code_cell("pd.read_csv(ROOT/'reports/tables/phase2_algorithm_agreement.csv')"),
        nbf.v4.new_code_cell("display(Image(filename=ROOT/'reports/figures/phase2_algorithm_agreement.png', width=800))"),
        nbf.v4.new_code_cell("pd.read_csv(ROOT/'reports/tables/phase2_proxy_label_metrics.csv')"),
        nbf.v4.new_markdown_cell("نتیجه محافظه‌کارانه: ساختار واقعی اما هم‌پوشان است. k=2 مرجع عملیاتی است، k=3 پایدارترین bootstrap و انتخاب BIC است، و kهای 6 و 7 به‌ترتیب توسط elbow و Gap پیشنهاد می‌شوند. بنابراین فاز سوم از consensus و تحلیل پروفایل برای حل این اختلاف استفاده می‌کند."),
    ]
    out=ROOT/'notebooks/02_phase2_clustering_executed.ipynb'
    out.parent.mkdir(exist_ok=True)
    executed=NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.write(executed,out)
    return out

if __name__=='__main__': print(build())
