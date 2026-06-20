"""スクレイパー共通のHTTPユーティリティ"""


def apply_netkeiba_encoding(response) -> None:
    """netkeiba のレスポンスに正しい文字エンコーディングを設定する。

    netkeiba はドメインによって文字コードが異なる:
      - race.netkeiba.com (出馬表・オッズ): HTTPヘッダに charset=UTF-8 を明示 → 2026年にUTF-8へ移行済み
      - db.netkeiba.com (結果・レース一覧・血統): charset 無し → EUC-JP

    HTTPヘッダに charset があれば requests が自動設定した値（=サーバ申告）を尊重し、
    無ければ EUC-JP にフォールバックする。これで両ドメインに追従できる。

    ※ encoding を "EUC-JP" 固定にすると、UTF-8 へ移行したページで全文字化けする。
    """
    content_type = response.headers.get("Content-Type", "").lower()
    if "charset=" not in content_type:
        response.encoding = "EUC-JP"
    # charset がヘッダにある場合は requests が response.encoding を設定済みなので何もしない
