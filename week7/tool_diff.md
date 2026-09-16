## Third Tool: `get_definitions`

### Description
```
Resolve a defined contractual term to its full definition, following any references to schedules. Use this to find what a defined term means (e.g. 'Business Day', 'Service Period').
```

### Parameters
```json
{
  "term_name": {
    "type": "STRING",
    "description": "The defined term to look up"
  },
  "contract_version": {
    "type": "STRING",
    "description": "Which contract version's definitions to check",
    "enum": [
      "v1_master",
      "v2_amendment_1",
      "v3_amendment_2"
    ]
  }
}
```

### Diff vs existing tools

| Aspect | search_clause | get_effective_date | **get_definitions** (NEW) |
|--------|--------------|-------------------|-------------------------|
| Job | Find clause TEXT | Return DATES | Resolve DEFINED TERMS |
| Enum param | -- | contract_version Y | contract_version Y |
| Input | clause name (str) | version (enum) | term name (str) + version (enum) |
| Schedule follow | No | No | **Yes** -- auto-resolves |
| Overlap | None with other two | None with other two | None with other two |
