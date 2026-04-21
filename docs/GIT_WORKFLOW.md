# Git操作ガイド

このドキュメントは、このプロジェクトで日常的に使うGit操作の手順をまとめたものです。
Claude Code のUIで「PRを作成」ボタンや「↑1」などが出た時の対処も含みます。

---

## 目次

1. [基本の流れ（変更をGitHubに保存する）](#1-基本の流れ変更をgithubに保存する)
2. [Claude Code UIの表示の意味](#2-claude-code-uiの表示の意味)
3. [コミットメッセージの書き方](#3-コミットメッセージの書き方)
4. [よく使うコマンド一覧](#4-よく使うコマンド一覧)
5. [よくあるトラブルと対処](#5-よくあるトラブルと対処)
6. [運用のコツ](#6-運用のコツ)

---

## 1. 基本の流れ（変更をGitHubに保存する）

コードを変更した後、GitHubに保存するまでの手順は4ステップです。

```bash
# ステップ1: 現状確認
git status

# ステップ2: 全変更をステージング
git add -A

# ステップ3: コミット
git commit -m "feat: 機能の簡潔な説明"

# ステップ4: リモートへプッシュ
git push origin master
```

この4コマンドを順に実行するだけです。以下、各ステップの詳細です。

### ステップ1: 何が変わったか把握する（git status / git diff）

```bash
git status           # ファイル一覧を表示
git diff             # 実際の差分を表示
git diff --stat      # ファイルごとの変更量サマリー
```

`git status` の出力例:
```
Changes not staged for commit:
  modified:   src/features/build_features.py
  modified:   app_pages/predict.py

Untracked files:
  app_pages/note_article.py
```

- **modified**: 既存ファイルが変更された
- **Untracked files**: 新規作成ファイル（まだGitが追跡していない）

### ステップ2: ステージング（git add）

ステージング = 「このファイルをコミットに含めます」という印をつける作業。

```bash
git add -A                    # 全部含める（推奨）
git add ファイル名            # 特定ファイルだけ
git add src/features/         # フォルダごと
git add -p                    # 対話的に選ぶ（上級者向け）
```

### ステップ3: コミット（git commit）

ステージングされた変更に「名前」を付けて履歴に記録する。

```bash
git commit -m "feat: v6モデル追加"
```

複数行メッセージを書きたい場合:
```bash
git commit -m "feat: v6モデル追加

- 種牡馬・母父の特徴量を5種追加
- テストAUC 0.7905を達成"
```

### ステップ4: プッシュ（git push）

ローカルのコミットをGitHubに送る。

```bash
git push origin master
```

**コミットしただけではまだGitHubには反映されていません。必ずpushまでやる。**

---

## 2. Claude Code UIの表示の意味

サイドバー下部の表示パターンと対処法。

### パターンA: 「+xxxx -yy」と緑の数字が出ている
```
 master ← master    [+12587 -110] [PRを作成]
```

**意味**: ローカルmaster に未コミットの変更がある（working tree と HEAD の差分）

**対処**: `git add -A && git commit -m "..." && git push origin master`

### パターンB: 「↑1」や「↑3」が出ている
```
 master    [↑1]
```

**意味**: コミットは済んでいるが、まだpushしていないコミットが N 個ある

**対処**: `git push origin master`

### パターンC: 「↓2」が出ている
```
 master    [↓2]
```

**意味**: リモートに新しいコミットが2個ある（他のPC等でpushされた）

**対処**: `git pull origin master`（マージが必要）

### パターンD: 「PRを作成」ボタンが出ている
**意味**: 何かしらローカルに変更がある

**対処**: 「PRを作成」は押す必要なし。上記A/Bどちらかの方法で消えます。PRを本当に作りたい時だけ押します。

---

## 3. コミットメッセージの書き方

このプロジェクトは **Conventional Commits** 形式を採用しています。

### 基本形式
```
タイプ: 簡潔な説明（50文字以内）

詳細を箇条書き（必要な場合）
- 変更内容1
- 変更内容2
```

### タイプ一覧

| タイプ | 用途 | 例 |
|---|---|---|
| `feat:` | 新機能追加 | `feat: v6モデル追加` |
| `fix:` | バグ修正 | `fix: オッズ計算の丸め誤差を修正` |
| `refactor:` | リファクタ（機能変化なし） | `refactor: predict_featuresを関数分割` |
| `perf:` | 性能改善 | `perf: 特徴量計算をベクトル化` |
| `docs:` | ドキュメント修正 | `docs: READMEを更新` |
| `chore:` | 雑多な変更（依存更新等） | `chore: requirementsを更新` |
| `test:` | テスト追加/修正 | `test: スクレイパーのテスト追加` |
| `style:` | コードスタイル（空白等） | `style: インデント統一` |

### 良いメッセージの例

```
feat: noteプラットフォーム配布用の記事生成ページを追加

- 会場別にまとめたMarkdown記事を自動生成
- 本命馬の推奨理由を特徴量から自動抽出（最大5件）
- .md / .txt でダウンロード可能
- 重賞・特別戦のみに絞り込むオプションあり
```

### 避けたい例
```
update          # 何を？ なぜ？が分からない
修正             # 何を？
aaa             # 意味不明
```

---

## 4. よく使うコマンド一覧

### 状態確認
```bash
git status                    # ファイル変更状況
git status -s                 # 1行で表示
git diff                      # 変更内容（未ステージ分）
git diff --staged             # 変更内容（ステージ済み分）
git log --oneline -10         # 直近10コミット
git log -p ファイル名         # ファイルの履歴
```

### ステージング
```bash
git add ファイル名            # 特定ファイルをステージ
git add -A                    # 全部ステージ
git restore --staged ファイル # ステージから外す（変更は残る）
```

### コミット
```bash
git commit -m "メッセージ"    # コミット
git commit --amend            # 直前のコミットを修正（未pushの場合のみ）
```

### プッシュ/プル
```bash
git push origin master        # pushする
git pull origin master        # pullする
git fetch                     # リモートの情報だけ取得（マージしない）
```

### ブランチ（使う場合）
```bash
git branch                    # ブランチ一覧
git checkout -b feature/xxx   # 新ブランチ作成・切替
git checkout master           # masterに戻る
```

### 変更を取り消す
```bash
git restore ファイル名        # 未ステージの変更を捨てる
git restore --staged ファイル # ステージングだけ取り消し
git reset --soft HEAD~1       # 直前のコミットを取り消し（変更は残す）
git reset --hard HEAD         # ⚠️全ての未コミット変更を破棄（破壊的）
```

---

## 5. よくあるトラブルと対処

### 「間違った内容でコミットしてしまった」（まだpushしてない）

**メッセージだけ直したい**:
```bash
git commit --amend -m "新しいメッセージ"
```

**ファイル追加を忘れた**:
```bash
git add 忘れたファイル
git commit --amend --no-edit
```

**コミット自体をなかったことにしたい**（変更は残す）:
```bash
git reset --soft HEAD~1
```

### 「pushしようとしたら `rejected` と言われた」

リモートに新しいコミットがあるため発生。

```bash
git pull origin master         # まずリモートの変更を取り込む
# マージコミットが作られる or コンフリクトが出る
git push origin master         # 改めてpush
```

### 「pushした後に間違いに気づいた」

**最も安全な方法**: 修正用の新しいコミットを追加する

```bash
# ファイルを修正
git add -A
git commit -m "fix: 前のコミットのタイポ修正"
git push origin master
```

**⚠️ `git push --force` は避ける**（他の人の作業を消す可能性あり）

### 「一部のファイルだけコミットしたい」

```bash
git add ファイル1 ファイル2       # 必要なファイルだけステージ
git commit -m "..."
# 他のファイルは次回以降にコミット
```

### 「.envなど機密ファイルを間違ってコミットした」

1. すぐにそのファイルを `.gitignore` に追加
2. インデックスから削除（ファイル自体は残す）:
   ```bash
   git rm --cached .env
   git commit -m "chore: .envをgit管理外に"
   ```
3. 既にpushしている場合、**パスワード等は必ず変更**（履歴に残っているため）

### 「ブランチやコミットIDを間違えて消してしまった」

30日以内なら復元可能:
```bash
git reflog                    # 過去の操作履歴を表示
git checkout <commit-hash>    # そのコミット時点に戻る
```

---

## 6. 運用のコツ

### こまめにコミット

✅ **良い例**:
- 「v6モデル追加」 1コミット
- 「note記事機能追加」 1コミット
- 「バグ修正」 1コミット

❌ **悪い例**:
- 「今日の変更」（何でも詰め込み）

1コミット = 1つの論理的な変更単位 にする。後から履歴を辿る自分が読みやすい。

### push忘れに注意

コミットしただけでは自分のPCにしか残っていない。PCが壊れると全部消える。
**定期的に `git push origin master` を実行する習慣**をつける。

### 作業前に pull

複数PC（会社/家/ノート）で作業する場合、**作業開始時に必ず `git pull`**。
忘れるとコンフリクトの原因になる。

### .gitignore を活用

以下は絶対にコミットしない:
- 機密情報（`.env`, APIキー, パスワード）
- 大きなバイナリ（`.pkl`, `.db`, ログ）
- PC固有のファイル（`.venv/`, `.vscode/`）
- 自動生成ファイル（`__pycache__/`）

このプロジェクトの `.gitignore` にはすでに主要なものが登録済み。

### 困ったら実行前に聞く

`git reset --hard`, `git push --force`, `git branch -D` は**取り返しがつかない**ことがある。
不安な場合は実行前にClaudeや熟練者に確認する。

---

## 参考: このプロジェクトのリポジトリ情報

- **リモート**: `https://github.com/yuki7621/velocity-keiba.git`
- **メインブランチ**: `master`
- **コミット形式**: Conventional Commits（英語タイプ + 日本語説明）
