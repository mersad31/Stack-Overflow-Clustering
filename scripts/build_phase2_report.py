from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stackoverflow_clustering.config import load_config
from stackoverflow_clustering.reporting import PersianReport, add_project_cover


def build() -> Path:
    config = load_config(ROOT / "config.yaml")
    artifacts = ROOT / "artifacts"
    tables = ROOT / "reports" / "tables"
    figures = ROOT / "reports" / "figures"

    partition_manifest = json.loads(
        (artifacts / "phase2_partitioning_manifest.json").read_text(encoding="utf-8")
    )
    family_manifest = json.loads(
        (artifacts / "phase2_model_families_manifest.json").read_text(encoding="utf-8")
    )
    partition_scores = pd.read_csv(tables / "phase2_partitioning_scores.csv")
    gap = pd.read_csv(tables / "phase2_gap_statistic.csv")
    family_scores = pd.read_csv(tables / "phase2_model_family_scores.csv")
    bootstrap_k = pd.read_csv(tables / "phase2_bootstrap_k_stability.csv")
    proxies = pd.read_csv(tables / "phase2_proxy_label_metrics.csv")
    agreement = pd.read_csv(tables / "phase2_algorithm_agreement.csv")
    low_silhouette = pd.read_csv(tables / "phase2_low_silhouette_cases.csv")

    report = PersianReport("مقایسه و اعتبارسنجی خانواده‌های الگوریتم خوشه‌بندی")
    add_project_cover(report, "گزارش فاز دوم پروژه داده‌کاوی پیشرفته")
    report.add_toc()

    report.heading("چکیده اجرایی", 1)
    report.paragraph(
        f"در این فاز K-Means و MiniBatch K-Means روی {partition_manifest['rows']:,} پاسخ‌دهنده، ۵۰ مؤلفه PCA و "
        "ده مقدار k اجرا شدند. مقایسه بین‌خانواده‌ای نیز برای جلوگیری از سوگیری نمونه روی یک نمونه ثابت "
        f"{family_manifest['sample_size']:,}تایی انجام شد و {family_manifest['configurations_evaluated']} پیکربندی از "
        "پنج خانواده partitioning، سلسله‌مراتبی، چگالی‌محور، مدل‌مبنا و Spectral را پوشش داد."
    )
    report.paragraph(
        "شواهد انتخاب k هم‌جهت نیستند: Silhouette و جمع رتبه‌ها k=2، Kneedle مقدار 6، Gap 1-SE مقدار 7 و "
        "بیشترین پایداری bootstrap مقدار 3 را ترجیح می‌دهند. بهترین silhouette معتبر خانواده‌ها متعلق به GMM-tied با k=2 "
        "است، در حالی که BIC مدل GMM-full با k=3 را برمی‌گزیند. روش‌های چگالی‌محور در grid مستندشده راه‌حل چندخوشه‌ای "
        "معتبر تولید نکردند. این اختلاف، ساختار هم‌پوشان و چندمقیاسی داده را نشان می‌دهد و مبنای استفاده از consensus در فاز سوم است."
    )
    report.callout(
        "جمع‌بندی فاز",
        "MiniBatch K-Means با k=2 مرجع تمام‌داده‌ای و بازتولیدپذیر است، اما برچسب مرجع داده محسوب نمی‌شود. فاز سوم باید چند نامزد معتبر "
        "را در ماتریس co-association تجمیع کند و عدم‌قطعیت انتخاب k را در تفسیر نگه دارد.",
    )
    report.table(
        ["شاهد", "نتیجه"],
        [
            ["Silhouette / جمع رتبه", "k=2"],
            ["Elbow / Kneedle", "k=6"],
            ["Gap 1-SE", "k=7"],
            ["Bootstrap stability", "k=3"],
            ["بهترین family silhouette", "GMM-tied، k=2"],
            ["بهترین GMM BIC", "GMM-full، k=3"],
        ],
    )

    report.heading("۱. طراحی آزمایش و قرارداد مقایسه", 1)
    report.paragraph(
        "تمام الگوریتم‌ها روی X_pca حاصل از RobustScaler اجرا شدند. برای K-Means و MiniBatch از کل cohort استفاده شد؛ معیارهای "
        "پرهزینه روی نمونه ثابت ۵۰۰۰تایی محاسبه شدند. مقایسه خانواده‌ها روی نمونه مشترک ۳۰۰۰تایی با seed=242 انجام شد تا "
        "تفاوت امتیازها ناشی از تغییر مشاهده‌ها نباشد. برای هر پیکربندی runtime، تعداد خوشه واقعی، سهم نویز، کمترین و بیشترین سهم "
        "خوشه و معیارهای داخلی ثبت شد."
    )
    report.table(
        ["خانواده", "الگوریتم‌ها", "فرض هندسی"],
        [
            ["Partitioning", "K-Means، MiniBatch", "خوشه‌های نسبتا کروی و centroidمحور"],
            ["Hierarchical", "single، complete، average، Ward", "ساختار سلسله‌مراتبی و linkage"],
            ["Density", "DBSCAN، HDBSCAN، OPTICS", "چگالی متغیر، نویز و شکل نامنظم"],
            ["Model-based", "GMM با ۴ covariance", "توزیع گاوسی و عضویت نرم"],
            ["Graph-based", "Spectral nearest-neighbors", "ساختار غیرکوژ روی گراف شباهت"],
        ],
    )
    report.callout(
        "قاعده اعتبار",
        "پیکربندی فقط زمانی نامزد انتخاب است که بیش از یک خوشه تولید کند و هر خوشه دست‌کم ۲٪ نمونه مشترک را داشته باشد. "
        "این قاعده تقسیم single-linkage با ۲۹۹۹ عضو در برابر یک عضو را کنار گذاشت، هرچند silhouette ظاهری آن بسیار بالا بود.",
    )

    report.heading("۲. مطالعه تعیین تعداد خوشه", 1)
    report.heading("۲.۱. Elbow، Silhouette و معیارهای داخلی", 2)
    report.paragraph(
        "Inertia با افزایش k به‌طور یکنواخت کاهش می‌یابد و به‌تنهایی انتخاب قطعی نمی‌دهد. Kneedle شکست منحنی را در k=6 تشخیص داد. "
        "Silhouette، Davies–Bouldin، Calinski–Harabasz و Dunn تقریبی روی نمونه ارزیابی مشترک محاسبه شدند. بهترین Silhouette و "
        "جمع رتبه ترکیبی به k=2 رسید؛ این نتیجه یک تقسیم درشت را ترجیح می‌دهد."
    )
    top_partition = partition_scores.sort_values("silhouette", ascending=False).head(10)
    report.table(
        ["الگوریتم", "k", "Silhouette", "DB", "CH", "Dunn", "زمان"],
        [
            (
                row.algorithm,
                row.k,
                f"{row.silhouette:.4f}",
                f"{row.davies_bouldin:.3f}",
                f"{row.calinski_harabasz:.1f}",
                f"{row.dunn_approx:.4f}",
                f"{row.runtime_seconds:.2f}s",
            )
            for row in top_partition.itertuples()
        ],
    )
    report.figure(figures / "phase2_partitioning_selection.png", "شکل ۱ — شواهد انتخاب k برای K-Means و MiniBatch")
    report.heading("۲.۲. Gap Statistic", 2)
    gap_display = gap.copy()
    gap_display["gap"] = gap_display["gap"].map(lambda value: f"{value:.4f}")
    gap_display["gap_standard_error"] = gap_display["gap_standard_error"].map(lambda value: f"{value:.4f}")
    report.table(
        ["k", "Gap", "خطای معیار", "شرط 1-SE"],
        gap_display[["k", "gap", "gap_standard_error", "one_se_rule_satisfied"]].itertuples(index=False, name=None),
    )
    report.paragraph(
        "قاعده یک انحراف معیار نخستین k را انتخاب می‌کند که بهبود بعدی نسبت به عدم‌قطعیت برتری معناداری نداشته باشد؛ در این داده k=7 "
        "پیشنهاد شد. اختلاف این نتیجه با Silhouette نشان می‌دهد Gap به زیرساختارهای ریزتر نسبت به جدایش درشت حساس‌تر است."
    )
    report.figure(figures / "phase2_gap_statistic.png", "شکل ۲ — Gap Statistic و انتخاب بر اساس قاعده 1-SE")
    report.heading("۲.۳. پایداری k", 2)
    bootstrap_display = bootstrap_k.copy()
    for column in ["mean", "std", "min"]:
        bootstrap_display[column] = bootstrap_display[column].map(lambda value: f"{value:.4f}")
    report.table(
        ["k", "تکرار", "میانگین ARI", "انحراف معیار", "کمینه"],
        bootstrap_display[["k", "repeats", "mean", "std", "min"]].itertuples(index=False, name=None),
    )
    report.paragraph(
        "بیشترین میانگین ARI بازنمونه‌گیری برای k=3 به دست آمد. پایداری بالا به معنی تفسیر دامنه‌ای بهتر نیست، اما نشان می‌دهد "
        "تقسیم سه‌خوشه‌ای در برابر حذف تصادفی بخشی از داده مقاوم‌تر است. در انتخاب نهایی این شاهد کنار معیارهای هندسی و تفسیر قرار گرفت."
    )
    report.figure(figures / "phase2_bootstrap_k_stability.png", "شکل ۳ — پایداری bootstrap به تفکیک k")
    report.figure(figures / "phase2_bootstrap_coassociation.png", "شکل ۴ — احتمال هم‌خوشه‌شدن زوج‌ها در بازنمونه‌گیری bootstrap")

    report.heading("۳. روش‌های Partitioning", 1)
    report.paragraph(
        "K-Means با چند شروع مستقل و MiniBatch K-Means با minibatchهای مقیاس‌پذیر اجرا شدند. هر دو الگوریتم برای تمام kها همگرا شدند. "
        f"مرجع برنده MiniBatch K-Means با k={partition_manifest['winner']['k']} و silhouette="
        f"{partition_manifest['winner']['silhouette']:.4f} بود. مدل نهایی روی تمام ۶۰٬۰۲۳ ردیف برازش و هش آن در manifest ثبت شد."
    )
    report.paragraph(
        f"پایداری بیست seed این مرجع میانگین ARI={partition_manifest['seed_stability']['mean']:.3f} و انحراف معیار "
        f"{partition_manifest['seed_stability']['std']:.3f} داشت. بیست bootstrap نیز میانگین {partition_manifest['bootstrap_stability']['mean']:.3f} "
        "ایجاد کرد. پراکندگی این مقادیر نشان می‌دهد centroidهای درشت قابل بازیابی‌اند، اما نقاط مرزی و زیرساختارها می‌توانند جابه‌جا شوند."
    )
    report.figure(figures / "phase2_partitioning_stability.png", "شکل ۵ — توزیع پایداری seed و bootstrap برای مرجع partitioning")
    report.figure(figures / "phase2_partitioning_umap.png", "شکل ۶ — برچسب مرجع partitioning روی UMAP دوبعدی")

    report.heading("۴. روش‌های سلسله‌مراتبی", 1)
    report.paragraph(
        "چهار linkage روی نمونه مشترک بررسی شد. single نسبت به chaining آسیب‌پذیر بود و تقسیم بسیار نامتوازن تولید کرد. complete و "
        "average خوشه‌های فشرده‌تر ساختند، اما Ward با k=2 بهترین راه‌حل معتبر این خانواده و silhouette حدود ۰٫۲۱۲ را ارائه داد. "
        "cophenetic correlation برای سنجش میزان حفظ فاصله‌های زوجی و دو راهبرد برش maxclust و آستانه فاصله ثبت شدند."
    )
    report.figure(figures / "phase2_hierarchical_dendrograms.png", "شکل ۷ — دندروگرام‌های چهار linkage سلسله‌مراتبی")

    report.heading("۵. روش‌های چگالی‌محور", 1)
    report.paragraph(
        "DBSCAN با k-distance، HDBSCAN با condensed tree و OPTICS با reachability curve تنظیم شدند. هدف یافتن شکل‌های نامنظم و "
        "جداسازی نویز بود. در grid مستندشده هیچ روش چگالی‌محور راه‌حل چندخوشه‌ای معتبر با حداقل سهم ۲٪ تولید نکرد؛ برخی تنظیم‌ها "
        "یک خوشه غالب همراه با نویز و برخی دیگر تقسیم‌های بسیار ریز ساختند. بنابراین در محدوده تنظیمات بررسی‌شده، روش‌های چگالی‌محور برای این داده مناسب نبودند."
    )
    report.figure(figures / "phase2_k_distance.png", "شکل ۸ — منحنی k-distance برای تنظیم روش‌های چگالی‌محور")
    report.figure(figures / "phase2_hdbscan_condensed_tree.png", "شکل ۹ — نمودار condensed tree الگوریتم HDBSCAN")
    report.figure(figures / "phase2_optics_reachability.png", "شکل ۱۰ — نمودار reachability الگوریتم OPTICS")
    report.figure(figures / "phase2_density_tradeoff.png", "شکل ۱۱ — مصالحه جدایش خوشه و سهم نویز")

    report.heading("۶. مدل‌های آمیخته گاوسی", 1)
    report.paragraph(
        "GMM برای covarianceهای spherical، diagonal، tied و full و kهای ۲ تا ۸ اجرا شد. همه مدل‌ها AIC، BIC، همگرایی، "
        "lower bound و entropy انتساب نرم را گزارش کردند. GMM-tied با k=2 بهترین silhouette معتبر خانواده‌ها را داشت، در حالی که "
        "کمینه BIC برای GMM-full با k=3 رخ داد. این تفاوت میان جدایش هندسی و برازش احتمالاتی باید در تفسیر حفظ شود."
    )
    report.figure(figures / "phase2_gmm_information_criteria.png", "شکل ۱۲ — AIC و BIC برای ساختارهای covariance مختلف")

    report.heading("۷. Spectral Clustering", 1)
    report.paragraph(
        "Spectral Clustering با affinity نزدیک‌ترین همسایه روی نمونه مشترک اجرا شد تا ساختارهای غیرکوژ بررسی شوند. بهترین راه‌حل "
        "این خانواده k=3 با silhouette حدود ۰٫۱۲۱ بود؛ امتیاز آن از Ward و GMM پایین‌تر است، اما الگوی متفاوتش تنوع مجموعه مدل‌های "
        "فاز سوم را بیشتر می‌کند. انتخاب اعضای consensus فقط بر رتبه silhouette متکی نیست و تفاوت فرض‌های هر الگوریتم نیز مهم است."
    )

    report.heading("۸. مقایسه نهایی خانواده‌ها", 1)
    valid = (
        family_scores.dropna(subset=["silhouette"])
        .query("clusters_found > 1 and minimum_cluster_fraction >= 0.02")
        .sort_values("silhouette", ascending=False)
        .head(15)
    )
    report.table(
        ["خانواده", "الگوریتم", "k", "Silhouette", "DB", "CH", "کمترین سهم", "زمان"],
        [
            (
                row.family,
                row.algorithm,
                row.clusters_found,
                f"{row.silhouette:.4f}",
                f"{row.davies_bouldin:.3f}",
                f"{row.calinski_harabasz:.1f}",
                f"{100 * row.minimum_cluster_fraction:.1f}٪",
                f"{row.runtime_seconds:.2f}s",
            )
            for row in valid.itertuples()
        ],
    )
    report.figure(figures / "phase2_family_silhouette_comparison.png", "شکل ۱۳ — بهترین پیکربندی‌های معتبر هر خانواده")

    report.heading("۹. متغیرهای کمکی، توافق و تحلیل خطا", 1)
    report.heading("۹.۱. معیارهای متغیرهای کمکی", 2)
    report.paragraph(
        "DevType، EdLevel، Employment، RemoteWork و AISelect برچسب مرجع نیستند. NMI فقط نشان می‌دهد خوشه‌ها تا چه اندازه با "
        "هر متغیر شناخته‌شده هم‌راستا هستند. مقادیر پایین به‌معنای شکست clustering نیست؛ در واقع نشان می‌دهد تقسیم نهایی صرفا "
        "یک نسخه بازکدگذاری‌شده از نقش یا تحصیلات نیست."
    )
    report.table(
        ["Proxy", "NMI", "دسته", "تفسیر"],
        [
            (row.proxy, f"{row.normalized_mutual_information:.4f}", row.categories, row.interpretation)
            for row in proxies.itertuples()
        ],
    )
    report.heading("۹.۲. توافق الگوریتم‌ها", 2)
    agreement_display = agreement.sort_values("adjusted_rand_index", ascending=False)
    report.table(
        ["نامزد اول", "نامزد دوم", "ARI"],
        [
            (row.candidate_a, row.candidate_b, f"{row.adjusted_rand_index:.4f}")
            for row in agreement_display.itertuples()
        ],
    )
    report.figure(figures / "phase2_algorithm_agreement.png", "شکل ۱۴ — ماتریس توافق زوجی نامزدهای consensus")
    report.heading("۹.۳. نقاط کم‌silhouette", 2)
    report.paragraph(
        "پایین‌ترین silhouetteها به پاسخ‌هایی تعلق دارند که به خوشه خود نزدیک نیستند یا میان centroidها قرار گرفته‌اند. این نقاط "
        "برای تحلیل مرز، بازبینی breadth و کنترل وابستگی به تجربه ذخیره شدند. منفی‌بودن silhouette یک فرد به معنی خطای داده نیست؛ "
        "نشانه ابهام انتساب در یک ساختار پیوسته است."
    )
    report.table(
        ["ResponseId", "خوشه", "Silhouette", "نقش", "تجربه", "Breadth"],
        [
            (
                int(row.ResponseId),
                row.cluster,
                f"{row.silhouette:.4f}",
                row.DevType,
                row.ExperienceConsensus,
                row.TechnologyBreadth,
            )
            for row in low_silhouette.head(12).itertuples()
        ],
    )

    report.heading("۱۰. نتیجه و انتقال به فاز سوم", 1)
    report.paragraph(
        "یک پاسخ واحد از همه معیارها پشتیبانی نمی‌شود. MiniBatch K-Means با k=2 مرجع تمام‌داده‌ای است؛ Ward-k2 جدایش "
        "سلسله‌مراتبی معتبر، GMM-full-k3 برازش احتمالاتی مناسب و Spectral-k3 ساختار گرافی متفاوت فراهم می‌کنند. این چهار نامزد "
        "برای ساخت co-association انتخاب شدند. فاز سوم باید k را در فضای consensus دوباره ارزیابی، برچسب نمونه را به کل داده تعمیم، "
        "پروفایل‌ها و نمونه‌های مرزی را تفسیر و حساسیت نتیجه را گزارش کند."
    )
    report.callout(
        "جمع‌بندی انتخاب مدل",
        "فاز دوم یک برنده قطعی ندارد و خروجی آن مجموعه‌ای از نامزدهای مناسب همراه با عدم‌قطعیت انتخاب است. این اختلاف‌ها در "
        "انتخاب اعضای consensus فاز سوم در نظر گرفته می‌شوند.",
    )

    report.heading("پیوست — بازتولیدپذیری", 1)
    report.code_block(".\\venv\\Scripts\\python.exe -m scripts.run_phase2 --config config.yaml")
    report.paragraph(
        "تمام seedها، gridها، اندازه نمونه و قاعده حداقل سهم در config.yaml ثبت شده‌اند. برچسب‌ها، linkageها، مدل GMM، مدل "
        "partitioning، جدول کامل ۷۸ پیکربندی و شکل‌های تشخیصی در data/processed، artifacts و reports ذخیره شده‌اند."
    )

    output = ROOT / "reports" / "Phase2_Report_FA.docx"
    report.save(output)
    return output


if __name__ == "__main__":
    print(build())
