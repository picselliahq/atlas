# Atlas: Agentic Computer Vision

[![Tests Status](https://github.com/picselliahq/atlas/actions/workflows/tests.yml/badge.svg)](https://github.com/picselliahq/atlas/actions)
[![Build Status](https://github.com/picselliahq/atlas/actions/workflows/build.yml/badge.svg)](https://github.com/picselliahq/atlas/actions)
[![uv](https://img.shields.io/badge/uv-DE5FE9?logo=uv&logoColor=white)](https://github.com/astral-sh/uv)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://github.com/python/cpython)
[![FastAPI](https://img.shields.io/badge/FastAPI-009485?logo=fastapi&logoColor=white)](https://github.com/fastapi/fastapi)
[![Code Coverage](https://img.shields.io/codecov/c/github/picselliahq/atlas/main.svg?label=Coverage&logo=codecov&logoColor=white&labelColor=F01F7A)](https://codecov.io/gh/picselliahq/atlas)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> **Atlas** is an open‑source collection of AI agents built with
**[SmolAgents](https://github.com/huggingface/smolagents)**
> and **[Pydantic‑AI](https://github.com/pydantic/pydantic-ai)** that super‑charges your workflows with
_agentic_ data‑analysis super‑powers. Think of Atlas as your autonomous data‑science teammate that can read your
> datasets, run analytical recipes, and surface insights through natural‑language conversations.


---

## 🌎 Why Atlas ?

We spend the last years building [Picsellia](https://picsellia.com) to **help every computer vision teams sort, manage
and maintain their data and models at scale.**

As a reminder, the original product is an **end-to-end platform** ranging features from data collection and labeling to
model monitoring and feedback-loop.

As our platform already empowers our users by augmenting their assets (data, annotations, experiments) by giving them a
schema that we support and maintain, we felt that **the newest LLM capabilities could make sense of all this
now-structured data**.

### We built Atlas with this in mind, using the latest Agentic and LLM capabilities to use our data and models as a knowledge base to answer questions, run analysis and surface insights.

The version you can see now only runs against images and annotations (to find outliers or labeling issues for example)
but we plan on making it available so it will be able to:

- **Run actions on your behalf (create new dataset, run data processing jobs, tag data that seems outdated...)**
- **Analyse and understand your data (this is the current version)**
- **Compare it with training results and evaluation metrics, so you can really understand your model and data flaws**
- **Orchestrate and analyze new experiments**
- **Monitor your predictions and monitoring metrics to detect drifts and anomalies and allow you to react timely**

We are extending the MCP of Picsellia, so Atlas can communicate easily and interact with every Picsellia objects! This is done through our very own chat interface, directly in the product.



---

## ✨ Key Features

- **Advanced Analytics** – we built a library of analysis that can be run on any dataset and extended with your own
  recipes.
- **Strong Typing ⊕ Validation** – every schema is enforced via Pydantic‑AI models.
- **Multi‑Agent Coordination** – orchestrate chains of SmolAgents for complex tasks.
- **LLM‑Powered Insights** – results are summarised into actionable narratives using your configured LLM provider.
- **First‑Class Picsellia 🎞️** – natively connects to Picsellia to fetch assets (images, tensors, predictions, …) for
  analysis.

---

## Agents

### Data-centric analysis workflow

- **Image Quality** – blur detection,luminance and contrast outliers...
- **Annotation Quality** – Outliers detection, missing labels, duplicates, overlapping labels...

### Operator

- **API** - Use our SDK to retrieve basic information about your dataset and images
- **Tools** - Use tools to perform actions like tagging, dataset creation, removing images ...

### Dataset Doctor

- Chat with the computed report to ask questions about your dataset and get insights

---

## 🚀 Quick Start

1. Create an account on [Picsellia](https://app.picsellia.com/signup).
2. Go to the Sample Dataset or create your own.
3. Once in your Dataset, find the `Ask Atlas` button in the top-right of the window and click on it to open the chat.
4. You can either just start talking with the MCP server (basic APIs) or click on `Launch Analysis` to start computing a
   report using the Agents.
5. Go to the `Atlas` tab in your Dataset and check the results!

---

## 🗺️ Roadmap

📌 _Provisional and subject to community feedback._

| Milestone | Tentative ETA | Description                                                                       |
|-----------|---------------|-----------------------------------------------------------------------------------|
| **v1.0**  | 2025‑04       | Agents for Data Analysis based on Picsellia                                       |
| **v1.1**  | 2025‑04       | Standalone mode without Picsellia dependency and docs                             |
| **v1.2**  | 2025‑04       | Contribution guidelines on how to add your own analysis                           |
| **v1.3**  | 2025‑05       | Add MCP support for all Picsellia objects (experiments, models ...)               |
| **v1.4**  | 2025‑Q3       | Add native analysis for Model Training and Prediction Monitoring (with Picsellia) |

Help us shape the roadmap! Open a [discussion](https://github.com/picselliahq/atlas/discussions) or vote on issues ✨.

---

## 🤝 Contributing

We 💙 contributions!

So far, the repository is only intended to be hosted by Picsellia and used from Picsellia but our **really next step**
is to
give you the ability to run Atlas on your own machine and use it with your own datasets as image and labels folders.
This way you will be able to customize the Agents, the LLM used, and even the analysis performed!

By doing this you are going to augment your Computer Vision capabilities to the next level 😎

Contribution guidelines will be released with the standalone version of Atlas so stay tuned!

---

### Community Standards

By participating you agree to abide by our [Code of Conduct](.github/CODE_OF_CONDUCT.md).

---

## 📄 License

Distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgements

- [HuggingFace](https://huggingface.co/) for SmolAgents
- [Pydantic](https://pydantic.dev/) for the best data models in town
- The awesome [Picsellia](https://picsellia.com) community

---

## 💬 Support & Community

- GitHub [Discussions](https://github.com/picselliahq/atlas/discussions)
- Join our **#atlas** channel in the [Picsellia Slack](https://picsellia.com/community)
- Follow [@picsellia](https://www.linkedin.com/company/picsell-ia) on Linkedin for updates

_Atlas shoulders the heavy lifting—so your insights feel weightless._ 🪽
