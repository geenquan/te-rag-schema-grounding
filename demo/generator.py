"""Deterministic synthetic demo dataset: "Aurora Logistics".

Everything here is FICTIONAL and was written for this public release: table
names, column names, Chinese business descriptions, questions and SQL. No
enterprise schema, query, or annotation was copied or adapted. The scenario
deliberately uses Chinese questions so the original jieba tokenization and
keyword-pattern logic run on their real code path; demo/README.md provides
English glosses.

Why both scales:
* ``small`` — a readable bundle (~44 questions, 16 tables) that exercises
  schema loading, SQL AST parsing, B/C annotation generation, table/column
  retrieval, full-gold evaluation, both protocols and the analysis flows.
  It is too small for the fold-level statistics to mean anything; those
  pipeline steps report "not computed" rather than fabricating numbers.
* ``full`` — the same generator with parameterized variants (~96 questions)
  yielding enough records and target-table clusters to exercise the paired
  statistics path end to end. Synthetic statistics validate the MACHINERY
  only; they are not paper results and must never be presented as such.

The generator is fully deterministic (no RNG): the same scale always
produces byte-identical CSVs.
"""

import csv
import os
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------
# Fictional schema: (table, description, [(column, description, type), ...])
# --------------------------------------------------------------------------

SCHEMA: List[Tuple[str, str, List[Tuple[str, str, str]]]] = [
    ("wms_in_orders", "入库单主表",
     [("order_no", "入库单号", "varchar"), ("warehouse_code", "仓库编码", "varchar"),
      ("supplier_id", "供应商ID", "varchar"), ("inbound_date", "入库日期", "date"),
      ("order_status", "单据状态", "varchar"), ("creator", "创建人", "varchar")]),
    ("wms_in_lines", "入库单明细",
     [("line_id", "明细行号", "int"), ("order_no", "入库单号", "varchar"),
      ("sku_id", "商品ID", "varchar"), ("qty", "入库件数", "int"),
      ("unit_price", "不含税单价", "decimal"), ("line_amount", "行金额", "decimal")]),
    ("wms_out_orders", "出库单主表",
     [("order_no", "出库单号", "varchar"), ("warehouse_code", "仓库编码", "varchar"),
      ("customer_id", "客户ID", "varchar"), ("outbound_date", "出库日期", "date"),
      ("order_status", "单据状态", "varchar")]),
    ("wms_out_lines", "出库单明细",
     [("line_id", "明细行号", "int"), ("order_no", "出库单号", "varchar"),
      ("sku_id", "商品ID", "varchar"), ("qty", "出库件数", "int"),
      ("weight_kg", "毛重公斤", "decimal")]),
    ("wms_stock", "库存余额表",
     [("warehouse_code", "仓库编码", "varchar"), ("sku_id", "商品ID", "varchar"),
      ("stock_qty", "可用库存件数", "int"), ("locked_qty", "锁定件数", "int"),
      ("updated_at", "更新时间", "datetime")]),
    ("wms_stock_moves", "库存变动流水",
     [("move_id", "流水号", "bigint"), ("warehouse_code", "仓库编码", "varchar"),
      ("sku_id", "商品ID", "varchar"), ("move_type", "变动类型", "varchar"),
      ("qty", "变动件数", "int"), ("move_date", "变动日期", "date")]),
    ("base_warehouse", "仓库档案",
     [("warehouse_code", "仓库编码", "varchar"), ("warehouse_name", "仓库名称", "varchar"),
      ("city", "所在城市", "varchar"), ("region", "所属大区", "varchar"),
      ("capacity_pallet", "库容板位数", "int")]),
    ("base_customer", "客户档案",
     [("customer_id", "客户ID", "varchar"), ("customer_name", "客户名称", "varchar"),
      ("level", "客户等级", "varchar"), ("city", "客户所在城市", "varchar")]),
    ("base_supplier", "供应商档案",
     [("supplier_id", "供应商ID", "varchar"), ("supplier_name", "供应商名称", "varchar"),
      ("city", "供应商所在城市", "varchar")]),
    ("base_sku", "商品档案",
     [("sku_id", "商品ID", "varchar"), ("sku_name", "商品名称", "varchar"),
      ("category", "品类", "varchar"), ("unit", "计量单位", "varchar")]),
    ("tms_shipment", "运单表",
     [("shipment_no", "运单号", "varchar"), ("customer_id", "客户ID", "varchar"),
      ("orig_warehouse", "发货仓库编码", "varchar"), ("dest_city", "目的城市", "varchar"),
      ("pieces", "运单件数", "int"), ("freight", "运费金额", "decimal"),
      ("ship_date", "发货日期", "date")]),
    ("tms_delivery", "签收记录",
     [("delivery_id", "签收记录号", "bigint"), ("shipment_no", "运单号", "varchar"),
      ("signer", "签收人", "varchar"), ("sign_time", "签收时间", "datetime"),
      ("sign_status", "签收状态", "varchar")]),
    ("fin_freight_bill", "运费结算单",
     [("bill_no", "结算单号", "varchar"), ("shipment_no", "运单号", "varchar"),
      ("settle_amount", "结算金额", "decimal"), ("settle_date", "结算日期", "date"),
      ("settle_status", "结算状态", "varchar")]),
    ("fin_receivable", "应收账款",
     [("ar_id", "应收流水号", "bigint"), ("customer_id", "客户ID", "varchar"),
      ("bill_month", "账期月份", "varchar"), ("ar_amount", "应收金额", "decimal"),
      ("received_amt", "已收金额", "decimal")]),
    ("rpt_daily_stock", "库存日报",
     [("stat_date", "统计日期", "date"), ("warehouse_code", "仓库编码", "varchar"),
      ("total_qty", "当日总库存件数", "int"), ("sku_count", "在库商品种数", "int")]),
    ("rpt_weekly_stock", "库存周报",
     [("week_start", "周起始日期", "date"), ("warehouse_code", "仓库编码", "varchar"),
      ("total_qty", "周库存件数", "int"), ("sku_count", "周在库商品种数", "int")]),
    ("rpt_monthly_revenue", "月度营收报表",
     [("month", "统计月份", "varchar"), ("city", "城市", "varchar"),
      ("revenue", "营业收入", "decimal"), ("freight_income", "运费收入", "decimal")]),
]

# --------------------------------------------------------------------------
# Questions: (question_zh, sql, primary_table, annotated_select_fields)
#
# SQL is MySQL dialect and every referenced column exists in SCHEMA, so the
# AST full gold comes out verified_complete — EXCEPT one deliberately
# broken record (references a column missing from the schema) that
# demonstrates the needs_review gold status and its exclusion from the
# valid denominator. Several queries alias columns with business words that
# differ from the schema comments (e.g. a Chinese business-word alias for the
# qty column), giving the
# direct-annotation miner TERM material; similar column names across tables
# (qty/pieces/stock_qty) give the constrained transfer candidate material.
# --------------------------------------------------------------------------

BASE_QUESTIONS: List[Tuple[str, str, str, str]] = [
    # --- wms_in_orders / wms_in_lines (inbound) ---
    ("2024年1月各仓库的入库件数",
     "SELECT o.warehouse_code, SUM(l.qty) AS 入库件数 FROM wms_in_lines l "
     "JOIN wms_in_orders o ON l.order_no = o.order_no "
     "WHERE o.inbound_date BETWEEN '2024-01-01' AND '2024-01-31' "
     "GROUP BY o.warehouse_code",
     "wms_in_lines", "warehouse_code|qty"),
    ("上个月入库金额最高的供应商",
     "SELECT o.supplier_id, SUM(l.line_amount) AS 入库金额 FROM wms_in_lines l "
     "JOIN wms_in_orders o ON l.order_no = o.order_no "
     "WHERE o.inbound_date >= '2024-05-01' AND o.inbound_date < '2024-06-01' "
     "GROUP BY o.supplier_id ORDER BY SUM(l.line_amount) DESC LIMIT 1",
     "wms_in_orders", "supplier_id|line_amount"),
    ("尚未关闭的入库单有多少",
     "SELECT COUNT(*) AS 单据数量 FROM wms_in_orders "
     "WHERE order_status <> 'CLOSED'",
     "wms_in_orders", "order_status"),
    ("2024年二季度每张入库单的明细行数",
     "SELECT l.order_no, COUNT(*) AS 行数 FROM wms_in_lines l "
     "JOIN wms_in_orders o ON l.order_no = o.order_no "
     "WHERE o.inbound_date BETWEEN '2024-04-01' AND '2024-06-30' "
     "GROUP BY l.order_no",
     "wms_in_lines", "order_no"),
    ("各供应商的平均入库单价",
     "SELECT o.supplier_id, AVG(l.unit_price) AS 平均单价 FROM wms_in_lines l "
     "JOIN wms_in_orders o ON l.order_no = o.order_no "
     "GROUP BY o.supplier_id",
     "wms_in_lines", "supplier_id|unit_price"),
    ("2024年3月入库量按日汇总",
     "SELECT o.inbound_date, SUM(l.qty) AS 入库量 FROM wms_in_lines l "
     "JOIN wms_in_orders o ON l.order_no = o.order_no "
     "WHERE o.inbound_date BETWEEN '2024-03-01' AND '2024-03-31' "
     "GROUP BY o.inbound_date",
     "wms_in_lines", "inbound_date|qty"),

    # --- wms_out_orders / wms_out_lines (outbound) ---
    ("2024年6月各仓库的出库件数",
     "SELECT o.warehouse_code, SUM(l.qty) AS 出库件数 FROM wms_out_lines l "
     "JOIN wms_out_orders o ON l.order_no = o.order_no "
     "WHERE o.outbound_date BETWEEN '2024-06-01' AND '2024-06-30' "
     "GROUP BY o.warehouse_code",
     "wms_out_lines", "warehouse_code|qty"),
    ("客户C001上季度的出库单列表",
     "SELECT order_no, outbound_date FROM wms_out_orders "
     "WHERE customer_id = 'C001' AND outbound_date BETWEEN '2024-04-01' AND '2024-06-30'",
     "wms_out_orders", "order_no|outbound_date"),
    ("出库明细的平均毛重",
     "SELECT AVG(weight_kg) AS 平均毛重 FROM wms_out_lines",
     "wms_out_lines", "weight_kg"),
    ("2024年上半年出库量前五的仓库",
     "SELECT o.warehouse_code, SUM(l.qty) AS 出库量 FROM wms_out_lines l "
     "JOIN wms_out_orders o ON l.order_no = o.order_no "
     "WHERE o.outbound_date BETWEEN '2024-01-01' AND '2024-06-30' "
     "GROUP BY o.warehouse_code ORDER BY SUM(l.qty) DESC LIMIT 5",
     "wms_out_lines", "warehouse_code|qty"),
    ("状态为待拣货的出库单",
     "SELECT order_no, customer_id FROM wms_out_orders WHERE order_status = 'PICKING'",
     "wms_out_orders", "order_no|customer_id"),

    # --- wms_stock / wms_stock_moves ---
    ("仓库WH01当前的可用库存件数",
     "SELECT sku_id, stock_qty FROM wms_stock WHERE warehouse_code = 'WH01'",
     "wms_stock", "sku_id|stock_qty"),
    ("锁定件数超过一千的商品",
     "SELECT warehouse_code, sku_id, locked_qty FROM wms_stock "
     "WHERE locked_qty > 1000",
     "wms_stock", "warehouse_code|sku_id|locked_qty"),
    ("2024年5月的库存变动流水",
     "SELECT move_id, move_type, qty FROM wms_stock_moves "
     "WHERE move_date BETWEEN '2024-05-01' AND '2024-05-31'",
     "wms_stock_moves", "move_id|move_type|qty"),
    ("各仓库最近的库存更新时间",
     "SELECT warehouse_code, MAX(updated_at) AS 最近更新 FROM wms_stock "
     "GROUP BY warehouse_code",
     "wms_stock", "warehouse_code|updated_at"),
    ("盘点类型变动的总件数",
     "SELECT SUM(qty) AS 盘点件数 FROM wms_stock_moves WHERE move_type = 'COUNT'",
     "wms_stock_moves", "qty|move_type"),

    # --- base_* archives ---
    ("华东大区有哪些仓库",
     "SELECT warehouse_code, warehouse_name FROM base_warehouse WHERE region = '华东'",
     "base_warehouse", "warehouse_code|warehouse_name"),
    ("库容板位数最大的仓库",
     "SELECT warehouse_code, capacity_pallet FROM base_warehouse "
     "ORDER BY capacity_pallet DESC LIMIT 1",
     "base_warehouse", "warehouse_code|capacity_pallet"),
    ("VIP客户名单",
     "SELECT customer_id, customer_name FROM base_customer WHERE level = 'VIP'",
     "base_customer", "customer_id|customer_name"),
    ("杭州的客户数量",
     "SELECT COUNT(*) AS 客户数量 FROM base_customer WHERE city = '杭州'",
     "base_customer", "city"),
    ("上海的供应商有哪些",
     "SELECT supplier_id, supplier_name FROM base_supplier WHERE city = '上海'",
     "base_supplier", "supplier_id|supplier_name"),
    ("零食大类的商品列表",
     "SELECT sku_id, sku_name FROM base_sku WHERE category = '零食'",
     "base_sku", "sku_id|sku_name"),
    ("以箱为单位计量的商品种数",
     "SELECT COUNT(*) AS 商品种数 FROM base_sku WHERE unit = '箱'",
     "base_sku", "unit"),

    # --- tms_shipment / tms_delivery ---
    ("2024年6月发往北京的运单件数",
     "SELECT SUM(pieces) AS 运单件数 FROM tms_shipment "
     "WHERE dest_city = '北京' AND ship_date BETWEEN '2024-06-01' AND '2024-06-30'",
     "tms_shipment", "pieces|dest_city"),
    ("客户C002去年的运费总额",
     "SELECT SUM(freight) AS 运费 FROM tms_shipment "
     "WHERE customer_id = 'C002' AND ship_date BETWEEN '2023-01-01' AND '2023-12-31'",
     "tms_shipment", "customer_id|freight"),
    ("从仓库WH02发货的运单号",
     "SELECT shipment_no FROM tms_shipment WHERE orig_warehouse = 'WH02'",
     "tms_shipment", "shipment_no|orig_warehouse"),
    ("单票运费超过五千的运单",
     "SELECT shipment_no, freight FROM tms_shipment WHERE freight > 5000",
     "tms_shipment", "shipment_no|freight"),
    ("上周已签收的运单",
     "SELECT d.shipment_no, d.signer FROM tms_delivery d "
     "WHERE d.sign_status = 'SIGNED' AND d.sign_time >= '2024-06-10'",
     "tms_delivery", "shipment_no|signer"),
    ("签收人张三经手的签收记录",
     "SELECT delivery_id, sign_time FROM tms_delivery WHERE signer = '张三'",
     "tms_delivery", "delivery_id|sign_time"),
    ("拒收状态的运单号",
     "SELECT shipment_no FROM tms_delivery WHERE sign_status = 'REJECTED'",
     "tms_delivery", "shipment_no|sign_status"),

    # --- fin_freight_bill / fin_receivable ---
    ("2024年5月的运费结算金额",
     "SELECT SUM(settle_amount) AS 结算金额 FROM fin_freight_bill "
     "WHERE settle_date BETWEEN '2024-05-01' AND '2024-05-31'",
     "fin_freight_bill", "settle_amount|settle_date"),
    ("未结算的运费单",
     "SELECT bill_no, shipment_no FROM fin_freight_bill WHERE settle_status = 'OPEN'",
     "fin_freight_bill", "bill_no|shipment_no"),
    ("客户C003四月的应收金额",
     "SELECT ar_amount FROM fin_receivable "
     "WHERE customer_id = 'C003' AND bill_month = '2024-04'",
     "fin_receivable", "customer_id|ar_amount"),
    ("账期内未收齐的应收账款",
     "SELECT ar_id, ar_amount, received_amt FROM fin_receivable "
     "WHERE received_amt < ar_amount",
     "fin_receivable", "ar_id|ar_amount|received_amt"),

    # --- rpt_daily_stock / rpt_weekly_stock / rpt_monthly_revenue ---
    ("昨日各仓库的库存日报",
     "SELECT warehouse_code, total_qty FROM rpt_daily_stock WHERE stat_date = '2024-06-19'",
     "rpt_daily_stock", "warehouse_code|total_qty"),
    ("上月的仓别库存明细",
     "SELECT warehouse_code AS 仓别, total_qty AS 存量 FROM rpt_daily_stock "
     "WHERE stat_date >= '2024-05-01' AND stat_date < '2024-06-01'",
     "rpt_daily_stock", "warehouse_code|total_qty"),
    ("2024年5月各仓库的库存件数",
     "SELECT warehouse_code, total_qty FROM rpt_daily_stock "
     "WHERE stat_date BETWEEN '2024-05-01' AND '2024-05-31'",
     "rpt_daily_stock", "warehouse_code|total_qty"),
    ("2024年6月各仓库的库存件数",
     "SELECT warehouse_code, total_qty FROM rpt_weekly_stock "
     "WHERE week_start BETWEEN '2024-06-03' AND '2024-06-30'",
     "rpt_weekly_stock", "warehouse_code|total_qty"),
    ("第25周各仓库的在库商品种数",
     "SELECT warehouse_code, sku_count FROM rpt_weekly_stock "
     "WHERE week_start BETWEEN '2024-06-17' AND '2024-06-23'",
     "rpt_weekly_stock", "warehouse_code|sku_count"),
    ("在库商品种数最多的仓库",
     "SELECT warehouse_code, sku_count FROM rpt_daily_stock "
     "ORDER BY sku_count DESC LIMIT 1",
     "rpt_daily_stock", "warehouse_code|sku_count"),
    ("2024年5月各城市的运费收入",
     "SELECT city, freight_income FROM rpt_monthly_revenue WHERE month = '2024-05'",
     "rpt_monthly_revenue", "city|freight_income"),
    ("营收超过一百万的城市",
     "SELECT city, revenue FROM rpt_monthly_revenue WHERE revenue > 1000000",
     "rpt_monthly_revenue", "city|revenue"),

    # --- cross-table joins (multi-table gold) ---
    ("客户C001的运单及签收情况",
     "SELECT s.shipment_no, d.sign_status FROM tms_shipment s "
     "JOIN tms_delivery d ON s.shipment_no = d.shipment_no "
     "WHERE s.customer_id = 'C001'",
     "tms_shipment", "shipment_no|sign_status"),
    ("各城市客户的应收总额",
     "SELECT c.city, SUM(r.ar_amount) AS 应收总额 FROM fin_receivable r "
     "JOIN base_customer c ON r.customer_id = c.customer_id "
     "GROUP BY c.city",
     "fin_receivable", "city|ar_amount"),
    ("仓库名称与库存件数",
     "SELECT w.warehouse_name, s.total_qty FROM rpt_daily_stock s "
     "JOIN base_warehouse w ON s.warehouse_code = w.warehouse_code "
     "WHERE s.stat_date = '2024-06-19'",
     "rpt_daily_stock", "warehouse_name|total_qty"),
    ("运费收入与实际结算的差额",
     "SELECT s.shipment_no, s.freight, b.settle_amount FROM tms_shipment s "
     "JOIN fin_freight_bill b ON s.shipment_no = b.shipment_no",
     "fin_freight_bill", "shipment_no|freight|settle_amount"),

    # --- near-duplicate pair: exercises duplicate-group fold safety ---
    ("2024年1月各仓库的出库件数",
     "SELECT o.warehouse_code, SUM(l.qty) AS 出库件数 FROM wms_out_lines l "
     "JOIN wms_out_orders o ON l.order_no = o.order_no "
     "WHERE o.outbound_date BETWEEN '2024-01-01' AND '2024-01-31' "
     "GROUP BY o.warehouse_code",
     "wms_out_lines", "warehouse_code|qty"),
    ("2024年2月各仓库的出库件数",
     "SELECT o.warehouse_code, SUM(l.qty) AS 出库件数 FROM wms_out_lines l "
     "JOIN wms_out_orders o ON l.order_no = o.order_no "
     "WHERE o.outbound_date BETWEEN '2024-02-01' AND '2024-02-29' "
     "GROUP BY o.warehouse_code",
     "wms_out_lines", "warehouse_code|qty"),

    # --- deliberately broken record (column absent from the schema) ---
    ("各仓库的体积使用率",
     "SELECT warehouse_code, used_volume FROM rpt_daily_stock",
     "rpt_daily_stock", "warehouse_code|used_volume"),
]

# Parameterized variants for scale=full: (template, months) — each expansion
# keeps the masked-token structure distinct from its siblings by varying the
# aggregate or attribute, not just the constant.
_FULL_TEMPLATES = [
    ("{m}各仓库的入库件数",
     "SELECT o.warehouse_code, SUM(l.qty) AS 入库件数 FROM wms_in_lines l "
     "JOIN wms_in_orders o ON l.order_no = o.order_no "
     "WHERE o.inbound_date BETWEEN '{d1}' AND '{d2}' "
     "GROUP BY o.warehouse_code", "wms_in_lines", "warehouse_code|qty"),
    ("{m}入库金额最高的供应商",
     "SELECT o.supplier_id, SUM(l.line_amount) AS 入库金额 FROM wms_in_lines l "
     "JOIN wms_in_orders o ON l.order_no = o.order_no "
     "WHERE o.inbound_date BETWEEN '{d1}' AND '{d2}' "
     "GROUP BY o.supplier_id ORDER BY 入库金额 DESC LIMIT 1",
     "wms_in_orders", "supplier_id|line_amount"),
    ("{m}各仓库的出库件数",
     "SELECT o.warehouse_code, SUM(l.qty) AS 出库件数 FROM wms_out_lines l "
     "JOIN wms_out_orders o ON l.order_no = o.order_no "
     "WHERE o.outbound_date BETWEEN '{d1}' AND '{d2}' "
     "GROUP BY o.warehouse_code", "wms_out_lines", "warehouse_code|qty"),
    ("{m}发往广州的运单件数",
     "SELECT SUM(pieces) AS 运单件数 FROM tms_shipment "
     "WHERE dest_city = '广州' AND ship_date BETWEEN '{d1}' AND '{d2}'",
     "tms_shipment", "pieces|dest_city"),
    ("{m}的运费结算金额",
     "SELECT SUM(settle_amount) AS 结算金额 FROM fin_freight_bill "
     "WHERE settle_date BETWEEN '{d1}' AND '{d2}'",
     "fin_freight_bill", "settle_amount|settle_date"),
    ("{m}各城市的运费收入",
     "SELECT city, freight_income FROM rpt_monthly_revenue "
     "WHERE month = '{ym}'", "rpt_monthly_revenue", "city|freight_income"),
]

_MONTHS = ["2024年1月", "2024年2月", "2024年3月", "2024年4月", "2024年5月",
           "2024年6月", "2024年7月", "2024年8月"]
_MONTH_BOUNDS = {
    "2024年1月": ("2024-01-01", "2024-01-31", "2024-01"),
    "2024年2月": ("2024-02-01", "2024-02-29", "2024-02"),
    "2024年3月": ("2024-03-01", "2024-03-31", "2024-03"),
    "2024年4月": ("2024-04-01", "2024-04-30", "2024-04"),
    "2024年5月": ("2024-05-01", "2024-05-31", "2024-05"),
    "2024年6月": ("2024-06-01", "2024-06-30", "2024-06"),
    "2024年7月": ("2024-07-01", "2024-07-31", "2024-07"),
    "2024年8月": ("2024-08-01", "2024-08-31", "2024-08"),
}


def _full_questions():
    out = []
    for tpl, sql_tpl, table, fields in _FULL_TEMPLATES:
        for m in _MONTHS:
            d1, d2, ym = _MONTH_BOUNDS[m]
            out.append((tpl.format(m=m), sql_tpl.format(d1=d1, d2=d2, ym=ym),
                        table, fields))
    return out


def build_questions(scale: str = "small"):
    questions = list(BASE_QUESTIONS)
    if scale == "full":
        questions += _full_questions()
    return questions


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def generate(out_dir: str, scale: str = "small") -> Dict[str, str]:
    """Write field_schema.csv, table_schema.csv, qa_data.csv under out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    paths["field_csv"] = os.path.join(out_dir, "field_schema.csv")
    paths["table_csv"] = os.path.join(out_dir, "table_schema.csv")
    paths["qa_csv"] = os.path.join(out_dir, "qa_data.csv")

    with open(paths["table_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["table", "table_desc"])
        for table, desc, _ in SCHEMA:
            writer.writerow([table, desc])

    with open(paths["field_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["table", "field_name", "field_name_desc", "field_type"])
        for table, _, fields in SCHEMA:
            for name, desc, ftype in fields:
                writer.writerow([table, name, desc, ftype])

    questions = build_questions(scale)
    with open(paths["qa_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "sql", "table", "field"])
        for question, sql, table, fields in questions:
            writer.writerow([question, sql, table, fields])

    print(f"generated {len(questions)} questions, {len(SCHEMA)} tables "
          f"({scale}) under {out_dir}")
    return paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=os.path.join(os.path.dirname(__file__),
                                                     "data"))
    ap.add_argument("--scale", default="small", choices=["small", "full"])
    generate(ap.parse_args().output, ap.parse_args().scale)
