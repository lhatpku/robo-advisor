# questions.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass
class MCQuestion:
    id: str
    text: str
    options: List[str]
    guidance: str

@dataclass
class MCAnswer:
    selected_index: int
    selected_label: str
    raw_user_text: str

# EXACT copies of your spreadsheet content (including spacing/case/ellipses)
QUESTIONS: List[MCQuestion] = [
    MCQuestion(
        id="q1",
        text="How much emergency savings do you currently have set aside?",
        options=[
            "Less than 3 months of salary",
            "3-6 months of salary",
            "more than 6 months",
        ],
        guidance=(
            "Having an adequate emergency fund ensures you can cover unexpected expenses without liquidating your investments prematurely. Typically, 3–6 months of essential living expenses is recommended to maintain financial stability."
        ),
    ),
    MCQuestion(
        id="q2",
        text="What portion of your total investable assets does this managed account represent?",
        options=[
            "less than 25%",
            "25% to 50%",
            "more than 50%",
        ],
        guidance=(
            "Understanding how much of your overall wealth this account represents helps us determine how much risk is appropriate. If this is a small portion of your assets, you may tolerate higher risk; if it’s your main portfolio, a more balanced approach may be suitable."
        ),
    ),
    MCQuestion(
        id="q3",
        text="What is your total investment  horizon for this account?",
        options=[
            "less than 5 years",
            "5-10 year",
            "10-15 year",
            "15-20 year",
            "20-25 year",
            "25-30 year",
            "30 year +",
        ],
        guidance=(
            "Your investment time horizon—the number of years before you expect to withdraw funds—is a key factor in determining the right asset allocation. Longer horizons allow for more growth-oriented investments."
        ),
    ),
    MCQuestion(
        id="q4",
        text="How likely are you to make early withdrawals from this account?",
        options=[
            "No",
            "Less Likely",
            "Likely",
        ],
        guidance=(
            "Frequent or early withdrawals can affect your investment strategy. If withdrawals are likely, we may recommend maintaining a more liquid or conservative portfolio to avoid selling assets at unfavorable times."
        ),
    ),
    MCQuestion(
        id="q5",
        text="How would you describe your level of investment knowledge?",
        options=[
            "A little",
            "Normal ",
            "Expert",
        ],
        guidance=(
            "Your investment knowledge helps us tailor the advice and explanations you receive. It ensures that recommendations are communicated in a way that matches your familiarity with financial concepts."
        ),
    ),
    MCQuestion(
        id="q6",
        text="How do you value portfolio growth versus income guarantee",
        options=[
            "Value growth more",
            "Treat them equal",
            "Value income guarantee more",
        ],
        guidance=(
            "This indicates the investment objective as to grow assets or guarantee income (risky versus conservative), the answer will impact how the final portfolio derailed from the neutral equity calculated before"
        ),
    ),
    MCQuestion(
        id="q7",
        text="If a market crashes and your account value drops a lot, what would be your action item afterwards",
        options=[
            "I would continue investing the same way, I believe in the long term the market will bounce back",
            "I will investment  less than half of my account in a more conservative portfolio",
            "I will investment more than half of my account in a more conservative portfolio",
        ],
        guidance=(
            "This indicates when market crashes what actions the investor will take which will indicate "
            "his/her risk preference. "
        ),
    ),
]
