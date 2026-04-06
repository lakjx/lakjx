> Memory的存在，使得对用户和项目的定制化和了解程度越来越高。_记忆架构_——
```
┌─────────────────────────────────────────────────────────────┐
│  第3层：团队记忆（Team Memory）                               │
│  跨用户、按仓库   │ 服务端同步 │ REST API + 乐观锁           │
│  秘密扫描保护     │ 文件监听器 │ 30种凭证检测规则             │
├─────────────────────────────────────────────────────────────┤
│  第2层：持久记忆（Persistent Memory / extractMemories）       │
│  跨会话、按项目   │ 本地文件   │ 后台 Fork Agent 自动提取     │
│  4种类型分类      │ MEMORY.md 索引 │ AI 相关性召回            │
├─────────────────────────────────────────────────────────────┤
│  第1层：会话记忆（Session Memory）                             │
│  仅当前会话      │ 单个文件   │ 10个固定章节 │ 服务于压缩     │
└─────────────────────────────────────────────────────────────┘
```
> 核心文件：`src/memdir/`（8个文件）、`src/services/extractMemories/`（2个）、`src/services/SessionMemory/`（3个）、`src/services/teamMemorySync/`（5个）
* memory 的分类（写入）
  * 内容语义分类 (**AutoMemory**)：user / feedback / project / references
* 落盘 (本地和持久化)、更新和维护（写入）
* 加载和注入 agentic loop
  * 召回 -> 加载
* 对应的 system prompt
* llm-based
  * rule (memory 的幻觉)
---
### Case
* manifest (声明式 (declarative) 元数据文件，声明清单):
  * 它是对 memory 目录中已有记忆文件的“轻量目录摘要”，不是全文内容，也不是索引文件 MEMORY.md 本身。
  * 它由 `src/memdir/memoryScan.ts` 生成。代码先递归扫描 memory 目录 (`~/.claude/projects/<仓库标识>/memory/`) 下除 MEMORY.md 之外的 .md 文件 (每个文件承载一条或一组同主题的持久记忆。)，只读取前面的 frontmatter 和文件时间，抽出这几个字段。
* AutoMemory files (**持续记忆系统**):
  * memory topic files，每个文件承载一条或一组同主题的持久记忆。按语义主题命名 (topic-oriented naming。)，不按时间命名，先更新已有文件，不要重复创建 (prompt-based, extract memories agent)。
  * 典型命名：
    * user_role.md
    * feedback_testing.md
    * project_release_freeze.md
    * reference_linear_board.md
  * frontmatter：
    * name
    * description
    * type
```yaml
---
name: 用户偏好-简洁回复
description: 用户不希望在每次回复末尾加总结
type: feedback
---

不要在回复末尾总结刚做了什么，用户能看到 diff。

**Why:** 用户明确要求过"stop summarizing what you just did"
**How to apply:** 所有回复结束时，直接结束，不加回顾性总结。
```

* MEMORY.md
> MEMORY.md 是索引，不是记忆本身。纯 Markdown，无 frontmatter。每条一行，约 150 字符以内：
```text
- [用户偏好-简洁回复](feedback_concise.md) — 不在回复末尾加总结
- [项目-合并冻结](project_freeze.md) — 3月5日起移动端发布冻结
```
* **两步保存流程**：
1. 写入记忆文件（如 feedback_concise.md）
2. 在 MEMORY.md 中添加索引行
* **存储目录结构**：
```Plaintext
~/.claude/
  projects/
    -<myproject>/   ← sanitizePath(项目根目录)
      memory/                               ← AUTO_MEM_DIRNAME
        MEMORY.md                           ← 索引文件
        user_role.md                        ← 私有记忆文件
        feedback_testing.md
        project_deadline.md
        reference_linear.md
        team/                               ← 团队记忆子目录
          MEMORY.md                         ← 团队索引
          project_api_migration.md
          reference_oncall_board.md
        logs/                               ← KAIROS 每日日志
          2026/
            03/
              2026-03-31.md
```
---

### 分类

* `CLAUDE.md` 更像“长期指令/规则”
* `AutoMem` / `TeamMem` 更像“长期知识库”

#### 按来源/作用域分类
* `src/utils/claudemd.ts`
    * `src/utils/memory/types.ts`:
        * **Managed**: 系统级托管规则
        * **User**: 用户级 `~/.claude/CLAUDE.md`
        * **Project**: 仓库内 `CLAUDE.md`、`.claude/CLAUDE.md`、`.claude/rules/*.md`
        * **Local**: 私有项目级 `CLAUDE.local.md`
        * **AutoMem**: 自动记忆目录的 `MEMORY.md`
        * **TeamMem**: 团队共享记忆目录的 `MEMORY.md`

#### 按内容语义分类

> `AutoMem` / `TeamMem` 里的“真正记忆文件”不是随便写的，它们有强约束的 4 类 taxonomy，定义在 `src/memdir/memoryTypes.ts`：

1.  **user**: `user` 被定义为“关于用户是谁”的记忆，而不是项目事实：
    * 写入时，模型在 extraction prompt 里会被要求按这个语义选型。召回时，type 会出现在 memory manifest 里，帮助 relevance selector 判断。
    * `user` 本质上是“个体适配层”。它改变的是回答风格、解释粒度、默认协作方式。
2.  **feedback**: 是最“行为约束型”的 memory。它存的不是用户是谁，而是“你以后应该怎么做”。
    * feedback 不只记录负反馈，也记录正反馈。
        * “不要 mock DB” 是 feedback
        * “这个 bundled PR 方式对” 也是 feedback
    * `'{{memory content - for feedback/project types, structure as: rule/fact, then **Why:** and **How to apply:** lines}}'`
        * 因为这类记忆不是静态事实，而是可迁移规则。没有 Why，模型以后只能机械套用；有了 Why，它才知道边界条件。
        * 这也是 frontmatter 示例里专门强调 feedback/project 正文结构的原因。
3.  **project**: `project` 存的是“项目里那些不能从代码直接推出的上下文”。
    * 这类 memory 的核心是“非代码可推导”。
    * 比如：截止日期、merge freeze、某次重构背后的合规原因、某个事故的业务背景。这些东西即使读完整个 repo，也未必能推出。
    * 所以 project 并不是“项目知识大杂烩”，而是“代码外的项目现实”。
    * project 往往在 recall 阶段最有价值，因为用户问“为什么要改这个”“最近这个方向的约束是什么”时，代码本身不足以回答。
    * 但运行时代码对它的处理仍然是统一的：
        * 扫描 frontmatter
        * 读 description
        * 让 selector 判断是否相关
        * 注入内容
4.  **reference**: 不是存事实本身，而是存“去哪找事实”。
    * 典型内容：
        * 某个 Linear project 是什么
        * 哪个 Grafana dashboard 看什么
        * 哪个 Slack channel 存什么信息
* 每种类型要求的结构
    * feedback 和 project 类型必须包含 **Why:** 和 **How to apply:** 行
    * 相对日期必须转换为绝对日期
    * 硬编码了 6 类排除项（即使用户明确要求也不保存）——
      1. 代码模式、架构、文件路径、项目结构——可从代码推导
      2. Git 历史、最近变更——git log/git blame 是权威来源
      3. 调试方案或修复步骤——修复在代码中，commit message 有上下文
      4. CLAUDE.md 中已有的内容——不重复
      5. 临时任务细节：进行中的工作、当前对话上下文
      6. 如果用户要求保存 PR 列表或活动摘要，要追问"什么是令人意外的或不明显的？"——只保存那部分
---

如果把四类 memory 压缩成四个问题：
* **user**: 这个用户是谁？
    * 存“用户画像”
    * 不存“项目状态”
    * 不存“代码事实”
* **feedback**: 我以后该怎么做？
* **project**: 这个项目当前处在什么现实语境里？
* **reference**: 如果要查外部信息，我该去哪里？

---

**Taxonomy System Prompts**

```xml
<type>
  <name>user</name>
  <description>Contain information about the user's role, goals, responsibilities, and knowledge. Great user memories help you tailor your future behavior to the user's preferences and perspective. Your goal in reading and writing these memories is to build up an understanding of who the user is and how you can be most helpful to them specifically. For example, you should collaborate with a senior software engineer differently than a student who is coding for the very first time. Keep in mind, that the aim here is to be helpful to the user. Avoid writing memories about the user that could be viewed as a negative judgement or that are not relevant to the work you're trying to accomplish together.</description>
  <when_to_save>When you learn any details about the user's role, preferences, responsibilities, or knowledge</when_to_save>
  <how_to_use>When your work should be informed by the user's profile or perspective...</how_to_use>
</type>
```
> 存用户画像、不存项目状态、不存代码事实
```xml
<type>
  <name>feedback</name>
  <description>Guidance the user has given you about how to approach work — both what to avoid and what to keep doing...</description>
  <when_to_save>Any time the user corrects your approach ... OR confirms a non-obvious approach worked ... Include *why* so you can judge edge cases later.</when_to_save>
  <how_to_use>Let these memories guide your behavior so that the user does not need to offer the same guidance twice.</how_to_use>
  <body_structure>Lead with the rule itself, then a **Why:** line ... and a **How to apply:** line ...</body_structure>
  <examples></examples>
</type>
```
```xml
<type>
  <name>project</name>
  <description>Information that you learn about ongoing work, goals, initiatives, bugs, or incidents within the project that is not otherwise derivable from the code or git history...</description>
  <when_to_save>When you learn who is doing what, why, or by when...</when_to_save>
  <how_to_use>Use these memories to more fully understand the details and nuance behind the user's request and make better informed suggestions.</how_to_use>
  <body_structure>Lead with the fact or decision, then a **Why:** line ... and a **How to apply:** line ...</body_structure>
  <examples></examples>
</type>

<type>
  <name>reference</name>
  <description>Stores pointers to where information can be found in external systems. These memories allow you to remember where to look to find up-to-date information outside of the project directory.</description>
  <when_to_save>When you learn about resources in external systems and their purpose...</when_to_save>
  <how_to_use>When the user references an external system or information that may be in an external system.</how_to_use>
  <examples></examples>
</type>
```
---

#### 按运行期用途分类

* **SessionMemory** 在 `src/services/SessionMemory/`，是“当前会话摘要笔记”，服务于长会话和 compact，不是项目长期知识库。
* **AgentMemory** 在 `src/tools/AgentTool/agentMemory.ts`，是给子 agent/专用 agent 的持久记忆，按 user/project/local scope 分目录。

#### 注入方式分类

* **作为 system prompt 的规则文本**
    * 规则，不是事实，`## When to access memories` / `## Before recommending from memory`
* **作为前置 meta user message 的上下文文本**
    * `claudeMd`
    * 为什么不是直接塞进 system prompt
        * 更像“随会话变化的上下文”，不是稳定规则。
* **作为附件转成的 meta user message**，例如 `nested_memory`、`relevant_memories`
    * `nested_memory`
        * 从 CWD -> target file 路径上的 `CLAUDE.md`
        * `.claude/rules` 里的 conditional rules
    * `relevant_memories`: 围绕当前用户问题的语义召回

---

### 写入（落盘、更新与维护）与召回

* **Managed/User/Project/Local** 这套更多是“读取和发现”，核心在 `src/utils/claudemd.ts`:
    * 它不是主要的自动写入目标，更多是人维护、系统读取。
* **AutoMem / TeamMem**: 文件型知识库，这是 Claude Code 真正的长期记忆库。
    * **AutoMem**: `<memoryBase>/projects/<sanitized-project-root>/memory/`
    * **入口文件**: `MEMORY.md`
        * `MEMORY.md` 是索引，不是正文；正文在一个个 topic `.md` 文件里。扫描时也会显式跳过 `MEMORY.md`，只扫 topic 文件 frontmatter。
    * **TeamMem**: `AutoMem/team/`
#### 自动记忆提取（extractMemories）
```
用户输入 → 模型响应 → 工具执行 → 循环... → 模型最终响应（无工具调用）
                                                    ↓
                                              Stop Hooks 执行
                                                    ↓
                                        executeExtractMemories()  ← 即发即忘
```
* **自动写入主要由 `extractMemories.ts` 触发**，它不是主模型直接写，而是 **turn** 结束后 fork 一个 agent 去做总结提取。触发时机如上——
  * 它先扫描已有记忆，给 extraction agent 一个 manifest
  * 然后 fork agent 自己决定“更新旧文件还是写新文件”
  * 通过 `createAutoMemCanUseTool()` 严格限制权限:
      * 可读: Read/Grep/Glob/只读 Bash
      * 可写: 仅限 memory 目录内 Edit/Write
  * 成功提取后，将写入的文件路径列表作为 `SystemMemorySavedMessage` 追加到主对话中，这样主代理知道记忆已更新。
* 如果开启 KAIROS，新记忆不直接维护 `MEMORY.md`，而是写入按日期分层的 daily log；夜间再由 `/dream` 类流程归并成 topic files + `MEMORY.md`。这是另一种“先 append-only，再 consolidate”的维护模式。
#### AI 驱动的记忆召回（findRelevantMemories）
1. **召回流程**
```
用户发送消息
  ↓
scanMemoryFiles(memoryDir)
  ↓ 递归读取所有 .md 文件（排除 MEMORY.md）
  ↓ 解析前 30 行 frontmatter（description + type）
  ↓ 按 mtime 降序排序，上限 200 个文件
  ↓
过滤掉 alreadySurfaced（之前轮次已展示的路径）
  ↓
selectRelevantMemories(query, memories, signal, recentTools)
  ↓ 构建文本清单：每行 "- [type] filename (timestamp): description"
  ↓ 附加 "Recently used tools: ..." 部分
  ↓
sideQuery → Sonnet 模型
  ↓ 系统提示：SELECT_MEMORIES_SYSTEM_PROMPT
  ↓ 用户消息：Query + Available memories
  ↓ 结构化输出：{ selected_memories: string[] }
  ↓ max_tokens: 256
  ↓
返回最多 5 条 RelevantMemory（路径 + mtime）
```

2. **选择提示词的关键规则** 
> `findRelevantMemories.ts`
```typescript
const SELECT_MEMORIES_SYSTEM_PROMPT = `You are selecting memories that will be useful to Claude Code as it processes a user's query. You will be given the user's query and a list of available memory files with their filenames and descriptions.

Return a list of filenames for the memories that will clearly be useful to Claude Code as it processes the user's query (up to 5). Only include memories that you are certain will be helpful based on their name and description.
- If you are unsure if a memory will be useful in processing the user's query, then do not include it in your list. Be selective and discerning.
- If there are no memories in the list that would clearly be useful, feel free to return an empty list.
- If a list of recently-used tools is provided, do not select memories that are usage reference or API documentation for those tools ...`
```
- 返回最多 5 个文件名
- 要有选择性和辨别力；如果不确定，不要包含
- 如果提供了最近使用的工具列表：
  - 不要选择这些工具的使用参考/API 文档（已经在用了）
  - 仍然选择这些工具的警告/陷阱/已知问题
 
> `memoryTypes.ts`
```typescript
export const WHEN_TO_ACCESS_SECTION: readonly string[] = [
  '## When to access memories',
  '- When memories seem relevant, or the user references prior-conversation work.',
  '- You MUST access memory when the user explicitly asks you to check, recall, or remember.',
  '- If the user says to *ignore* or *not use* memory: proceed as if MEMORY.md were empty. Do not apply remembered facts, cite, compare against, or mention memory content.',
  MEMORY_DRIFT_CAVEAT,
]

export const TRUSTING_RECALL_SECTION: readonly string[] = [
  '## Before recommending from memory',
  '',
  'A memory that names a specific function, file, or flag is a claim that it existed *when the memory was written*. It may have been renamed, removed, or never merged. Before recommending it:',
  '',
  '- If the memory names a file path: check the file exists.',
  '- If the memory names a function or flag: grep for it.',
  '- If the user is about to act on your recommendation (not just asking about history), verify first.',
  '',
  '"The memory says X exists" is not the same as "X exists now."',
]
```
  * memory 里提到某个具体符号时，它只能被当作“历史线索”，不能被当作“当前事实”。
      * 也就是说，memory 说：有一个文件 `foo/bar.ts`、有一个函数 `getUserProfile()`、有一个 flag `ENABLE_X`，并不等于它们现在还存在。
  * 因为 memory 记录的是“写下这条记忆时的认知”，而代码库是持续变化的（memory 幻觉）。这个函数/文件/flag 可能后来：
      * 被重命名了
      * 被删除了
      * 从来没真正合并进当前分支
      * 只存在于过去某个分支或某次讨论里
  * 所以这段 prompt 是在明确要求模型：
      * memory 不是 source of truth
      * memory 是候选线索
      * 真正要给用户建议之前，必须回到当前代码/当前仓库状态去验证

3. **召回前的验证要求**
```text
// system prompt
You have a persistent, file-based memory system with two directories: a private directory at `/Users/me/.claude/projects/my-repo/memory/` and a shared team directory at `/Users/me/.claude/projects/my-repo/memory/team/`. Both directories already exist — write to them directly with the Write tool (do not run mkdir or check for their existence).

You should build up this memory system over time so that future conversations can have a complete picture of who the user is...

## When to access memories
- When memories (personal or team) seem relevant, or the user references prior work...
- You MUST access memory when the user explicitly asks you to check, recall, or remember.
- If the user says to ignore or not use memory: proceed as if MEMORY.md were empty...

## Before recommending from memory
A memory that names a specific function, file, or flag is a claim that it existed when the memory was written. It may have been renamed, removed, or never merged. Before recommending it:
- If the memory names a file path: check the file exists.
- If the memory names a function or flag: grep for it.
- If the user is about to act on your recommendation, verify first.
```
- 如果记忆提到一个文件路径 → 检查文件是否存在
- 如果记忆提到一个函数或标志 → grep 搜索它
- 如果用户即将根据你的建议行动 → 先验证
- "记忆说 X 存在"不等于"X 现在存在"
- 活动日志、架构快照是冻结在某个时间点的，优先使用 git log


4. **更新记忆和维护**
当前仓库真实做法是一个受限的 extraction subagent，请求内容来自 `buildExtractAutoOnlyPrompt` / `buildExtractCombinedPrompt`。
* **调用时机**
    * 每次 turn 结束；
* **system**: 继承父会话的 systemPrompt
    * 语言偏好
    * 工具使用原则
    * 安全规则
    * memory 的基础政策
    * CLAUDE.md 注入后的上下文
    * ...
* **user message**
```typescript
return [
  `You are now acting as the memory extraction subagent. Analyze the most recent ~${newMessageCount} messages above and use them to update your persistent memory systems.`,
  '',
  `Available tools: ${FILE_READ_TOOL_NAME}, ${GREP_TOOL_NAME}, ${GLOB_TOOL_NAME}, read-only ${BASH_TOOL_NAME} ...`,
  '',
  `You have a limited turn budget. ${FILE_EDIT_TOOL_NAME} requires a prior ${FILE_READ_TOOL_NAME} of the same file, so the efficient strategy is: turn 1 — issue all ${FILE_READ_TOOL_NAME} calls in parallel ...`,
  '',
  `You MUST only use content from the last ~${newMessageCount} messages to update your persistent memories. Do not waste any turns attempting to investigate or verify that content further ...`
]
```
**完整返回**

```json
{
  "model": "sonnet-class",
  "system": "You are selecting memories that will be useful ... Be selective and discerning. Prefer constraints, warnings, and project context over generic references.",
  "messages": [
    {
      "role": "user",
      "content": "Query: <user task>\n\nAvailable memories:\n- [feedback] feedback_testing.md ...\n- [project] release_freeze.md ...\n\nRecently used tools: Bash, ReadFile\n\nAlready surfaced: release_freeze.md"
    }
  ],
  "output_format": {
    "type": "json_schema",
    "schema": {
      "type": "object",
      "properties": {
        "selected_memories": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "path": { "type": "string" },
              "reason": { "type": "string" },
              "confidence": { "type": "number" }
            },
            "required": ["path", "reason"]
          }
        }
      },
      "required": ["selected_memories"]
    }
  }
}
```


5. **注入 agentic loop**

* **agentic loop**
    * system prompt: memory policy / how-to-use memory
    * user context: claudeMd
    * meta user messages / attachments: relevant_memories、nested_memory
* **让 agent 在行动时**：
    * 知道什么时候该信 memory
    * 知道什么时候该先验证
    * 把 memory 当成行动约束和上下文，而不是当成绝对事实库

* **claudeMd & currentDate**
    * 会被包装成一条 `<system-reminder>` 风格的 meta user message，放在所有对话前面。


**运行期注入结构示例：**

```json
{
  "role": "user",
  "content": "<system-reminder>\nMemory (saved 3 days ago):\n/.../feedback_testing.md:\n\n---\nname: ...\ndescription: ...\ntype: feedback\n---\nIntegration tests must hit a real database...\n</system-reminder>"
}

{
  "model": "main-agent-model",
  "system": [
    "<main system prompt>",
    "<memory policy: when to access, what not to save, how to use>",
    "<before recommending from memory: verify files/functions/flags first>"
  ],
  "messages": [
    {
      "role": "user",
      "content": "<system-reminder>\n# selected_memories\nMemory (saved 3 days ago): ...\n\nMemory (saved 12 days ago): ...\n</system-reminder>"
    },
    {
      "role": "user",
      "content": "<real user request>"
    }
  ],
  "tools": ["ReadFile", "EditFile", "Shell", "Glob", "Grep"]
}
```
### 会话记忆（Session Memory）
#### 与持久记忆的关键区别
| 维度 | 会话记忆 | 持久记忆 |
| :--- | :--- | :--- |
| 生命周期 | 当前会话 | 跨会话永久 |
| 文件数量 | 1个 | 多个 |
| 触发频率 | 每次后采样（需满足阈值） | 每轮查询结束 |
| 主要用途 | 服务于上下文压缩 | 跨会话知识保留 |
| 记忆提取者 | Fork Agent（仅 FileEdit） | Fork Agent（Read/Write/Edit/Grep） |
| 格式 | 10个固定章节 | 自由格式 + frontmatter |
#### 触发阈值
```typescript
// 默认配置（可通过 GrowthBook tengu_sm_config 远程调整）
const defaults = {
  minimumMessageTokensToInit: 10_000,   // 上下文达到 10K tokens 才激活
  minimumTokensBetweenUpdate: 5_000,    // 每增长 5K tokens 更新一次
  toolCallsBetweenUpdates: 3,           // 且至少 3 次工具调用
}

// 触发条件：
// (token 阈值满足 AND 工具调用阈值满足)
// OR (token 阈值满足 AND 最后一轮助手消息无工具调用)
```
#### 会话记忆模板（10 个固定章节）
```typescript
# Session Memory

## Session Title
_Brief description of what this session is about_

## Current State
_What is the current status of the work_

## Task specification
_Detailed description of the current task requirements_

## Files and Functions
_Key files and functions being worked on_

## Workflow
_Steps being followed or processes in use_

## Errors & Corrections
_Errors encountered and how they were resolved_

## Codebase and System Documentation
_Important codebase patterns, conventions, and system behavior_

## Learnings
_Insights gained during this session_

## Key results
_Important outputs, measurements, or achievements_

## Worklog
_Chronological log of actions taken_
```
**大小限制**
```typescript
const MAX_SECTION_LENGTH = 2_000      // 每个章节最大 2K tokens
const MAX_TOTAL_SESSION_MEMORY_TOKENS = 12_000  // 整个文件最大 12K tokens
```
#### 与压缩的集成
Session Memory 是上下文压缩的优先路径（参见 `sessionMemoryCompact.ts`）：
- 自动压缩触发时，优先尝试用 Session Memory 作为摘要
- 保留最近 10K-40K tokens 的消息
- 比传统压缩快得多（不调用 LLM），且摘要质量更可预测
