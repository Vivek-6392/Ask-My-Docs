"""
scripts/seed_eval_docs.py
Seeds the vector store + BM25 index with synthetic documents
that answer the questions in evaluation/eval_dataset.json.
Used by CI before running the RAGAS evaluation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import AppConfig
from retrieval.pipeline import RAGPipeline

SEED_DOCS = [
    {
        "text": (
            "Digital products and downloads are non-refundable once the download "
            "has been initiated or the license key has been revealed. Exceptions apply "
            "only in cases of verified technical failure confirmed by our support team "
            "within 48 hours of purchase."
        ),
        "metadata": {"source": "refund_policy.pdf", "page": 3},
    },
    {
        "text": (
            "To reset your password, navigate to the login page and click the "
            "'Forgot Password' link beneath the sign-in form. Enter your registered "
            "email address. A password-reset link will be sent within 5 minutes. "
            "The link expires after 30 minutes for security."
        ),
        "metadata": {"source": "user_guide.pdf", "page": 12},
    },
    {
        "text": (
            "Enterprise tier system requirements: minimum 16 GB RAM, quad-core CPU "
            "at 2.4 GHz or faster. Supported operating systems: Ubuntu 20.04 LTS or "
            "later, Windows Server 2019 or later, macOS 12 Monterey or later."
        ),
        "metadata": {"source": "enterprise_setup.pdf", "page": 7},
    },
    {
        "text": (
            "All reports and datasets within the platform can be exported in CSV "
            "format. Navigate to the Dashboard, select the desired report, then click "
            "Settings > Export > CSV. Exports are available for the Pro and Enterprise "
            "tiers only."
        ),
        "metadata": {"source": "dashboard_guide.pdf", "page": 21},
    },
    {
        "text": (
            "Service Level Agreement (SLA): The platform guarantees 99.9% monthly "
            "uptime excluding scheduled maintenance windows. If uptime falls below "
            "this threshold, affected customers receive service credits prorated to "
            "the duration of the outage, as described in Section 8 of the Terms."
        ),
        "metadata": {"source": "sla_terms.pdf", "page": 2},
    },
]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config = AppConfig.from_env()
    pipeline = RAGPipeline(config)
    n = pipeline.add_documents(SEED_DOCS)
    print(f"✅ Seeded {n} documents into vector + BM25 stores.")


if __name__ == "__main__":
    main()
