# Directory Search with OpenRouter Decisions

Disclaimer: This project is fully AI-generated and should not be used in production.

This program uses Jev to scan directories of text files and source code to locate passages that match natural language criteria. It communicates with the OpenRouter Decisions API to evaluate text chunks against user-provided conditions.

## Overview

The search engine operates in two phases. First, it discovers source code and plain text files while ignoring build directories, package locks, and binary assets. It analyzes candidate files using keyword density to identify the most relevant files. It splits these files into text chunks and prioritizes chunks containing matching terms.

The scanner dispatches requests in parallel to the OpenRouter Decisions endpoint using a persistent HTTP connection pool. When the high priority phase finishes, results are printed to the terminal and written to a text file immediately. The engine then continues scanning the remaining lower priority files, updating both the console and the output file with any additional matches.

## Requirements

The project uses Python 3.10 or newer. Standard library modules handle process execution, parsing, and file handling. The requests library is utilized for HTTP connection pooling when available.

You need an OpenRouter API key with access to the decision models.

## Usage

Set your OpenRouter API key as an environment variable or pass it directly to the command line.

Exporting the environment variable:

```bash
export OPENROUTER_API_KEY="your-api-key"
python3 jev_search.py /path/to/folder "criteria expression"
```

Passing the key directly:

```bash
python3 jev_search.py /path/to/folder "criteria expression" --api-key "your-api-key"
```

## Criteria Syntax

Criteria expressions support AND, OR, parentheses, and quoted strings. AND operators take precedence over OR operators. Parentheses can be used to group expressions.

Matching multiple requirements:

```bash
python3 jev_search.py ./open-webui "handles token streaming AND updates chat interface"
```

Matching alternative requirements:

```bash
python3 jev_search.py ./open-webui "Server-Sent Events streaming OR WebSocket chat connection"
```

Grouping with parentheses:

```bash
python3 jev_search.py ./open-webui 'streaming AND ("chat UI" OR "markdown message renderer")'
```

## Command Line Options

The first positional argument specifies the folder path to scan. The second positional argument defines the criteria expression.

The `--workers` option sets the number of concurrent worker threads. The default value is 30.

The `--threshold` option sets the minimum confidence score required to consider a chunk a match. The default value is 0.7.

The `--top-files` option sets how many files are evaluated during the initial high priority phase. The default value is 25.

The `--output` option sets the file path for saving scan results. The default value is search_results.txt.

The `--chunk-size` option sets the maximum character length for each evaluated chunk. The default value is 2000.

The `--extensions` option restricts scanning to specific file extensions, such as `.svelte` or `.py`.

## Output

Matches are reported in order of score. Each result includes the file path, the aggregate score, and the text content of the matching passage. Results from the high priority phase appear first, followed by results from the low priority phase. All reported matches are saved to the designated text file.
