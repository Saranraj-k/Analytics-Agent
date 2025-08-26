from typing import Optional, Dict, Any
import pandas as pd
import uuid
import os
from pathlib import Path


from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.messages import HumanMessage


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIR = PROJECT_ROOT / "static" / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


class DataframeAgent:
    def __init__(self, llm, df: pd.DataFrame):
        self.llm = llm
        self.df = df
        # allow_dangerous_code True so the agent can execute pandas/numpy/matplotlib snippets when needed
        # verbose=False for cleaner responses
        self.agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            allow_dangerous_code=True,
            verbose=False,
        )

    def _save_last_figure(self) -> Optional[str]:
        """
        If Matplotlib has created any figures, save the last one to static/generated/
        and return a relative URL path suitable for front-end consumption.
        """
        try:
            figs = plt.get_fignums()
            if not figs:
                return None
            # get the last figure produced
            last_fig_num = figs[-1]
            fig = plt.figure(last_fig_num)
            fname = f"plot-{uuid.uuid4().hex}.png"
            out_path = GENERATED_DIR / fname
            fig.savefig(out_path, bbox_inches="tight")
            # Close the figure to free memory
            plt.close(fig)
            # Return a web-accessible relative URL (Flask serves /static/)
            return f"/static/generated/{fname}"
        except Exception as e:
            # If saving fails, return None (we still return textual output)
            return None

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Run a query against the agent and return a structured response.
        If the agent generated a matplotlib plot, also return an image_url.
        """
        try:
            # Clear any existing figures to ensure we capture only new ones
            plt.close('all')
            result = self.agent.invoke({"input": query})
            # result is typically a dict with 'output' key
            output_text = result.get("output", "")
            image_url = self._save_last_figure()
            response = {"ok": True, "answer": output_text}
            if image_url:
                response["image_url"] = image_url
            return response
        except Exception as e:
            # Try capture any figure even on failure (useful when code partially ran)
            image_url = self._save_last_figure()
            resp = {"ok": False, "error": str(e)}
            if image_url:
                resp["image_url"] = image_url
            return resp
