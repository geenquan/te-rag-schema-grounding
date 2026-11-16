# Synthetic Demo Dataset — "Aurora Logistics"

Everything in `demo/` is **fictional and was written for this public
release**: table names, column names, Chinese business descriptions,
questions, and SQL. No enterprise schema, query, annotation, or identifier
was copied or adapted. The questions are Chinese on purpose, so the original
jieba tokenization and keyword-pattern logic run on their real code path;
English glosses are provided below.

## Scenario

A logistics company ("Aurora Logistics") runs a warehouse-management
system (WMS), a transport-management system (TMS), finance ledgers, and
reporting tables. Analysts ask business questions in Chinese; the reference
SQL is MySQL.

## Scale

* `small` (default, committed): 49 questions over 17 tables. Exercises
  schema loading, SQL AST parsing, B/C annotation generation with
  provenance, table/column retrieval, full-gold evaluation, both protocols,
  and every analysis flow. It is intentionally too small for fold-level
  statistics to mean anything; the tooling reports what is and is not
  computable rather than fabricating numbers.
* `full` (`--scale full`): parameterized monthly variants (~93 questions)
  for exercising the paired-statistics path end to end on deterministic
  data. Synthetic statistics validate the machinery only.

The generator is fully deterministic (no RNG): the same scale produces
byte-identical CSVs.

## Contents

| File | Content |
|---|---|
| `generator.py` | deterministic generator (tables, questions, SQL) |
| `data/field_schema.csv` | table, field_name, field_name_desc, field_type |
| `data/table_schema.csv` | table, table_desc |
| `data/qa_data.csv` | question, sql, table, field (original annotation) |
| `data/full_gold_v2.jsonl` | AST-derived full gold (scripts/build_full_gold.py) |

One question (`q048` in the committed order) deliberately references a
column that does not exist in the schema; its gold status is `needs_review`
and it is excluded from the valid evaluation denominator — demonstrating
honest gold-status handling rather than silent exclusion.

## Table glossary

| Table | Description | English gloss |
|---|---|---|
| wms_in_orders / wms_in_lines | 入库单主表 / 入库单明细 | inbound order header / lines |
| wms_out_orders / wms_out_lines | 出库单主表 / 出库单明细 | outbound order header / lines |
| wms_stock | 库存余额表 | stock balance |
| wms_stock_moves | 库存变动流水 | stock movement journal |
| base_warehouse / base_customer / base_supplier / base_sku | 档案表 | master data: warehouses / customers / suppliers / SKUs |
| tms_shipment | 运单表 | shipment |
| tms_delivery | 签收记录 | delivery/sign-off records |
| fin_freight_bill | 运费结算单 | freight settlement |
| fin_receivable | 应收账款 | accounts receivable |
| rpt_daily_stock / rpt_weekly_stock | 库存日报 / 库存周报 | daily / weekly stock reports |
| rpt_monthly_revenue | 月度营收报表 | monthly revenue report |

## Why the near-twin report tables

`rpt_daily_stock` and `rpt_weekly_stock` intentionally share column names,
comments, and question phrasings ("2024年5月各仓库的库存件数" /
"2024年6月各仓库的库存件数" — "stock quantity per warehouse in May 2024 /
June 2024"), while only the daily report's columns get direct TERM evidence
(aliases like 仓别 "warehouse label" and 存量 "on-hand amount" in
"上月的仓别库存明细"). This gives the constrained one-hop transfer (C)
realistic candidate material: the weekly report's identical columns are
sparse targets, and shared slot-normalized templates provide the template
overlap the admission gates require. The demo then shows accepted C records
with full provenance (run with `--diagnostics`, then
`terag_release analyze --analysis transfer`).

## Sample questions (with glosses)

| Question | Gloss | Target table |
|---|---|---|
| 2024年1月各仓库的入库件数 | inbound quantity per warehouse, Jan 2024 | wms_in_lines |
| 上个月入库金额最高的供应商 | top supplier by inbound amount last month | wms_in_orders |
| 客户C001上季度的出库单列表 | C001's outbound orders last quarter | wms_out_orders |
| 锁定件数超过一千的商品 | SKUs with >1000 locked units | wms_stock |
| 华东大区有哪些仓库 | warehouses in the East region | base_warehouse |
| 2024年6月发往北京的运单件数 | shipment pieces to Beijing, Jun 2024 | tms_shipment |
| 客户C003四月的应收金额 | C003's receivable amount in April | fin_receivable |
| 昨日各仓库的库存日报 | yesterday's stock daily report | rpt_daily_stock |
| 各城市客户的应收总额 | receivables per customer city | fin_receivable |
| 2024年2月各仓库的出库件数 | outbound quantity per warehouse, Feb 2024 | wms_out_lines |

The near-duplicate pair ("...1月各仓库的出库件数" / "...2月各仓库的出库件数")
exercises the duplicate-group fold safety rule.
