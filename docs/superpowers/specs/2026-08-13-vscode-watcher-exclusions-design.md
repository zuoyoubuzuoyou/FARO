# VS Code 文件监听排除设计

## 目标

降低 FARO 工作区对 Linux `inotify` 监听额度的占用，消除 VS Code 无法监视文件更改的警告。

## 设计

新建 `.vscode/settings.json`，仅通过 `files.watcherExclude` 排除以下大型数据目录：

- `partnr-planner/data`
- `EMOS/data`

排除项使用工作区相对 glob，并匹配目录内全部内容。该配置不使用 `files.exclude` 或 `search.exclude`，所以目录仍然可见、可搜索、可打开和编辑；仅外部文件变化可能需要手动刷新。

## 验证

- 配置文件是合法 JSON。
- 两个目标目录均有独立的 `files.watcherExclude` 条目。
- 不添加其他排除项，也不修改项目运行配置。
