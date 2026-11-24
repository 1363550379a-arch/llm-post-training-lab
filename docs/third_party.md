# Distribution scope

The proposed MIT license covers original project code and documentation. Model weights, training records containing dataset text, Python packages and raw dataset files are obtained separately. This package contains training scripts, aggregate numeric logs and evaluation identifiers.

|External component|Upstream notice|Use in this release|
|---|---|---|
|[Qwen2.5-0.5B model card](https://huggingface.co/Qwen/Qwen2.5-0.5B)|Apache-2.0|Download by model identifier; weights excluded.|
|[GSM8K dataset card](https://huggingface.co/datasets/openai/gsm8k)|MIT|Load using datasets; original questions and solutions excluded.|
|[TRL UltraFeedback binary dataset card](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized)|This specific card does not declare a license.|Referenced by identifier; no source records or response text included. Review upstream rights before redistributing data.|
|[TRL](https://github.com/huggingface/trl), [PEFT](https://github.com/huggingface/peft), [Transformers](https://github.com/huggingface/transformers)|External Python dependencies|Installed from upstream; no vendored package directories.|

The final repository visibility and project license are subject to the author's publication approval.
