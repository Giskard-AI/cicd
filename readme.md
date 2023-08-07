# Giskard CI/CD runner (WIP)

## Overview

The idea is to have a common CI/CD core that can interface with different input sources (loaders) and output destinations (reporters).

The **core** is responsible for running the tests and generating a report.

The **loaders** are responsible for loading the model and dataset, wrapped as Giskard objects, from a given source (for example the HuggingFace hub, a Github repository, etc.).

The **reporters** are responsible for sending the report to the appropriate destination (e.g. a comment to a Github PR, a HuggingFace discussion, etc.).


### Tasks

Task could be data objects containig all the information needed to run a CI/CD pipeline. For example:

```json
{
    "loader_id": "huggingface",
    "model_id": "distilbert-base-uncased",
    "dataset_id": "sst2",
    "loader_args": {
        "dataset_split": "validation",
    },
    "reporter_id": "huggingface_discussion",
    "reporter_args": {
        "discussion_id": 1234,
    }
}
```

or


```json
{
    "loader_id": "github",
    "model_id": "my.package::load_model",
    "dataset_id": "my.package::load_test_dataset",
    "loader_args": {
        "repository": "My-Organization/my_project",
        "branch": "dev-test2",
    },
    "reporter_id": "github_pr",
    "reported_args": {
        "repository": "My-Organization/my_project",
        "pr_id": 1234,
    }
}
```

These tasks may be generated by a watcher (e.g. a Github action, a HuggingFace webhook, etc.) and put in a queue. The CI/CD runner will then pick them up and run the pipeline.

Otherwise, a single task can be created to run a single-shot Github action, without queueing.


### CI/CD Core

In pseudocode, the CI/CD core could look like this:

```python
task = get_task_from_queue_or_envirnoment()

loader = get_loader(task.loader_id)
gsk_model, gsk_dataset = loader.load_model_dataset(
    task.model_id,
    task.dataset_id,
    **task.loader_args,
)

runner = PipelineRunner()
report = runner.run(gsk_model, gsk_dataset)

reporter = get_reporter(task.reporter_id)
reporter.push_report(report, **task.reporter_args)
```

## Prototype

Current implementation has a `huggingface` loader and a prototype can be run from the command line:

```bash
$ python cli.py --loader huggingface --model distilbert-base-uncased-finetuned-sst-2-english --dataset_split validation --output demo_report.html
```

This will launch a pipeline that will load the model and dataset from the HuggingFace hub, run the scan and generate a report in HTML format (for now).
