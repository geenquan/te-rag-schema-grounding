"""
SQL role parser.

Parses SQL through sqlglot's AST to identify the role each column plays in
the statement.

Capabilities:
- Multiple dialects (mysql by default; falls back to hive for ODPS/Hive-style
  comments and functions; sqlite configurable for BIRD)
- Multi-table SQL (JOIN / subqueries / derived tables / CTEs); gold tables
  include every physical table
- Scoped alias resolution (T1/T2, derived-table aliases, CTE names; each
  subquery keeps an independent scope instead of one global alias pool)
- Unqualified column resolution (unique visible table / unique schema match;
  multi-table ambiguity is recorded explicitly, never silently attributed to
  the first table)
- Three-state star handling: SELECT * and t.* expand via the schema; a star
  inside an aggregate is recorded but not expanded
- Parse completeness is flagged independently: ast_parsed does not imply every
  column resolved; unresolved / ambiguous / star_expansions items are all
  persisted, so a "success with zero columns" result cannot mask a failure
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

try:
    import sqlglot
    from sqlglot import exp
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False
    print("Warning: sqlglot not available, falling back to regex-based parsing")


class _Unresolved(Exception):
    """A column could not be resolved to a physical table column."""


class _Ambiguous(Exception):
    """A column name exists in several visible tables (no unique owner)."""

    def __init__(self, column: str, candidates):
        super().__init__(column)
        self.column = column
        self.candidates = list(candidates)


@dataclass
class FieldRoleInfo:
    """Role information for one table column."""
    table: str
    field: str
    roles: Set[str] = field(default_factory=set)  # SELECT, WHERE, JOIN, GROUP_BY, ORDER_BY, HAVING
    role_weights: Dict[str, float] = field(default_factory=dict)

    def add_role(self, role: str, weight: float = 1.0):
        """Add a role, keeping the maximum weight seen for it."""
        self.roles.add(role)
        self.role_weights[role] = max(self.role_weights.get(role, 0), weight)

    def get_max_weight(self, role_weights: Dict[str, float]) -> float:
        """Return the maximum role weight across this column's roles."""
        if not self.roles:
            return 1.0
        return max(role_weights.get(r, 1.0) for r in self.roles)


class SQLRoleParser:
    """
    SQL role parser.

    Uses sqlglot AST parsing to resolve, per SELECT scope, which physical
    table each column belongs to and which role it plays.

    Supported roles:
    - SELECT / WHERE / JOIN / GROUP_BY / ORDER_BY / HAVING / FROM

    Usage:
        parser = SQLRoleParser(dialect='sqlite')
        result = parser.parse(sql, schema_columns={'frpm': {'County Name', ...}})
        print(result.tables)        # all physical tables
        print(result.field_roles)   # {table_name: {field_name: FieldRoleInfo}}
        print(result.unresolved)    # unresolved items (gold builders must inspect these)
        print(result.ambiguous)     # columns ambiguous across tables
    """

    def __init__(self, default_role_weights: Dict[str, float] = None,
                 dialect: str = 'mysql', mode: str = 'ast'):
        """
        Initialize the parser.

        Args:
            default_role_weights: default role weights
            dialect: SQL dialect (mysql by default for PowerGrid; hive is tried
                automatically for ODPS-style comments/functions)
            mode: 'ast' (default; full AST-based parsing) or 'regex' (legacy
                  regex fallback for diagnostic comparison only; not a
                  complete parse)
        """
        self.default_role_weights = default_role_weights or {
            'SELECT': 1.2,
            'WHERE': 2.0,
            'JOIN': 1.8,
            'GROUP_BY': 1.5,
            'ORDER_BY': 1.3,
            'HAVING': 1.4,
            'FROM': 1.0,
        }
        self.dialect = dialect
        if mode not in ('ast', 'regex'):
            raise ValueError(f"unknown parse mode: {mode}")
        self.mode = mode
        self._scopes_cache: Dict[int, 'SQLRoleParser._Scope'] = {}

    def parse(self, sql: str, expected_table: str = None,
              dialect: str = None,
              schema_columns: Optional[Dict[str, Set[str]]] = None) -> 'ParseResult':
        """
        Parse a SQL statement.

        Args:
            sql: the SQL statement
            expected_table: expected table name (kept for API compatibility;
                scope-based resolution no longer uses it to guess ownership)
            dialect: SQL dialect (overrides the default)
            schema_columns: physical schema {table: {column, ...}}, used for
                unqualified-column disambiguation, derived-table/CTE output
                matching, and star expansion. Required for complete gold
                construction; when omitted, unqualified columns in
                multi-table scopes are recorded as unresolved rather than
                guessed.

        Returns:
            ParseResult (ast_parsed marks successful AST parsing; for
            column-level completeness check unresolved / ambiguous /
            star_expansions)
        """
        if self.mode == 'regex' or not SQLGLOT_AVAILABLE:
            result = self._parse_with_regex(sql, expected_table)
            result.ast_parsed = False
            reason = ('explicit regex mode (diagnostic comparison)' if self.mode == 'regex'
                      else 'sqlglot-not-installed')
            result.unresolved.append({'item': reason,
                                      'kind': 'parser_missing',
                                      'reason': 'regex fallback; not a complete AST parse'})
            return result

        dialect = dialect or self.dialect

        # Dialect chain: configured dialect -> hive (ODPS-style
        # comments/functions) -> mysql (BIRD backticks)
        tried = []
        for d in [dialect] + [x for x in ('hive', 'mysql') if x != dialect]:
            tried.append(d)
            result = self._parse_with_sqlglot(sql, expected_table, d, schema_columns)
            if result.success:
                return result
        result = self._parse_with_regex(sql, expected_table)
        result.ast_parsed = False
        result.unresolved.append(
            {'item': f'sqlglot-failed:{",".join(tried)}', 'kind': 'parse_error',
             'reason': (result.error or '')[:300]})
        return result

    # ------------------------------------------------------------------ scopes

    class _Scope:
        """Scope of one SELECT: visible physical-table aliases, derived-table
        aliases, and CTEs."""

        def __init__(self, node, parent):
            self.node = node      # exp.Select
            self.parent = parent  # enclosing _Scope (correlated subqueries see outer tables)
            self.table_aliases: Dict[str, str] = {}       # alias/table name (lower) -> physical table
            self.derived_aliases: Dict[str, object] = {}  # derived-table/CTE alias -> inner query

    def _build_scopes(self, parsed) -> Dict[int, '_Scope']:
        """Build a scope for every exp.Select (parent link + FROM/JOIN sources + visible CTEs)."""
        selects = list(parsed.find_all(exp.Select))
        parent_select = {}
        for s in selects:
            p = s.parent
            while p is not None and not isinstance(p, exp.Select):
                p = p.parent
            parent_select[id(s)] = p

        # CTE definitions (WITH clause): name -> inner query
        cte_defs: Dict[str, object] = {}
        for cte in parsed.find_all(exp.CTE):
            cte_defs[cte.alias_or_name.lower()] = cte.this

        scopes: Dict[int, 'SQLRoleParser._Scope'] = {}
        for s in selects:
            parent_scope = scopes.get(id(parent_select[id(s)])) \
                if parent_select.get(id(s)) is not None else None
            sc = self._Scope(s, parent_scope)

            sources = []
            # sqlglot >=30 uses 'from_' (avoiding the Python keyword); older versions use 'from'
            frm = s.args.get('from_') or s.args.get('from')
            if frm is not None:
                # All direct sources of the From clause (normally only 'this';
                # parenthesized JOINs split into several)
                for v in frm.args.values():
                    vs = v if isinstance(v, list) else [v]
                    for x in vs:
                        if isinstance(x, exp.Expression):
                            sources.append(x)
            for join in (s.args.get('joins') or []):
                sources.append(join.this)
            for src in sources:
                self._register_source(src, sc, cte_defs)
                # Pre-order walk of the FROM tree (stops at Select boundaries):
                # aliased Subquery/Table under parenthesized JOINs or unusual
                # wrappers must remain visible in this scope
                self._collect_nested_sources(src, sc, cte_defs, depth=0)

            scopes[id(s)] = sc
        return scopes

    def _collect_nested_sources(self, node, sc: '_Scope',
                                cte_defs: Dict[str, object], depth: int):
        """Walk the FROM source tree registering aliased Subquery/Table nodes; never descend into inner Selects."""
        if node is None or depth > 12:
            return
        if isinstance(node, exp.Select):
            return  # Select boundary: the inner scope handles its own sources
        if isinstance(node, exp.Table):
            self._register_source(node, sc, cte_defs)
            return
        if isinstance(node, exp.Subquery):
            if node.alias:
                self._register_source(node, sc, cte_defs)
            for child in node.args.values():
                vs = child if isinstance(child, list) else [child]
                for c in vs:
                    if isinstance(c, exp.Expression):
                        self._collect_nested_sources(c, sc, cte_defs, depth + 1)
            return
        # Join / other container nodes: keep descending
        for child in node.args.values():
            vs = child if isinstance(child, list) else [child]
            for c in vs:
                if isinstance(c, exp.Expression):
                    self._collect_nested_sources(c, sc, cte_defs, depth + 1)

    def _register_source(self, src, sc: '_Scope', cte_defs: Dict[str, object]):
        """Register one FROM/JOIN source (physical table / derived table / CTE reference / parenthesized JOIN)."""
        if src is None:
            return
        if isinstance(src, exp.Table):
            name = src.name
            if not name:
                return
            if name.lower() in cte_defs:
                sc.derived_aliases[name.lower()] = cte_defs[name.lower()]
                return
            alias = (src.alias or name).lower()
            sc.table_aliases[alias] = name
            sc.table_aliases[name.lower()] = name
            return

        if isinstance(src, exp.Subquery):
            alias = (src.alias or '')
            # Recursively strip anonymous parenthesis wrappers; the real alias
            # may sit on an inner Subquery
            core, depth = src.this, 0
            while isinstance(core, exp.Subquery) and depth < 8:
                if not alias and core.alias:
                    alias = core.alias
                if not core.alias:
                    core = core.this
                    depth += 1
                else:
                    break

            if not alias and isinstance(core, exp.Join):
                # Parenthesized JOIN wrapper: FROM ( (…) a JOIN (…) b ) —
                # expand transparently
                self._register_source(core.this, sc, cte_defs)
                for j in (core.args.get('joins') or []):
                    self._register_source(j.this, sc, cte_defs)
                for deep in list(core.find_all(exp.Subquery)):
                    self._register_source(deep, sc, cte_defs)
                for deep in list(core.find_all(exp.Table)):
                    self._register_source(deep, sc, cte_defs)
                return

            inner = core.this if isinstance(core, exp.Subquery) else core
            key = alias.lower() if alias else f'__anon_derived_{len(sc.derived_aliases)}'
            sc.derived_aliases[key] = inner
            return

        if hasattr(src, 'alias'):
            # Set operations such as UNION act as derived sources directly
            alias = (getattr(src, 'alias', '') or '')
            key = alias.lower() if alias else f'__anon_derived_{len(sc.derived_aliases)}'
            sc.derived_aliases[key] = src

    def _visible_physical_tables(self, sc: '_Scope') -> List[str]:
        """All physical tables visible along the scope chain (deduplicated, first-seen order)."""
        out, seen = [], set()
        s = sc
        while s is not None:
            for name in s.table_aliases.values():
                low = name.lower()
                if low not in seen:
                    seen.add(low)
                    out.append(name)
            s = s.parent
        return out

    def _schema_cols(self, table: str, schema_columns) -> Optional[Set[str]]:
        if not schema_columns:
            return None
        cols = schema_columns.get(table) or schema_columns.get(table.lower())
        if cols is None:
            return None
        return {str(c).lower() for c in cols}

    def _scope_for_node(self, node):
        """Return the scope of the nearest SELECT enclosing the node."""
        p = node.parent
        while p is not None:
            if isinstance(p, exp.Select):
                sc = self._scopes_cache.get(id(p))
                if sc is not None:
                    return sc
            p = p.parent
        return None

    def _resolve_in_derived(self, inner, column_name: str, schema_columns,
                            depth: int = 0) -> Optional[Set[Tuple[str, str]]]:
        """Locate the physical (table, column) set a column name maps to inside a derived-table/CTE output."""
        if depth > 8:
            return None
        if not isinstance(inner, exp.Select):
            # Set operations such as UNION: search both sides
            found = set()
            for side in (getattr(inner, 'this', None),
                         getattr(inner, 'expression', None)):
                if side is not None:
                    sub = self._resolve_in_derived(side, column_name,
                                                   schema_columns, depth + 1)
                    if sub:
                        found |= sub
            return found or None

        has_star = False
        for e in inner.expressions:
            if list(e.find_all(exp.Star)):
                has_star = True
                continue
            output_name = (e.alias_or_name or '').lower()
            if output_name and output_name == column_name.lower():
                phys = set()
                for col in e.find_all(exp.Column):
                    col_scope = self._scope_for_node(col)
                    if col_scope is None:
                        continue
                    try:
                        phys |= self._resolve_column(col, col_scope, schema_columns)
                    except (_Unresolved, _Ambiguous):
                        continue
                return phys or None
        if has_star:
            # No explicit output matched but the inner query contains a star:
            # look among the star-expanded physical columns
            phys = set()
            sc = self._scopes_cache.get(id(inner))
            if sc is None:
                return None
            for t in self._visible_physical_tables(sc):
                cols = self._schema_cols(t, schema_columns)
                if cols and column_name.lower() in cols:
                    phys.add((t, column_name))
            return phys or None
        return None

    def _resolve_column(self, col: 'exp.Column', sc: '_Scope',
                        schema_columns) -> Set[Tuple[str, str]]:
        """Resolve one column node -> set of physical (table, column) pairs.

        Qualified columns: walk the scope chain for an alias / derived table /
        CTE (inner scopes naturally shadow outer ones).
        Unqualified columns (local-first, matching SQL name resolution):
        single physical table in this scope -> assign; unique schema match ->
        assign; ambiguous in this scope -> record ambiguity; no match here ->
        try this scope's derived outputs; then repeat in enclosing scopes.

        Raises:
            _Unresolved / _Ambiguous
        """
        qualifier = (col.table or '').lower()
        name = col.name

        if qualifier:
            s = sc
            while s is not None:
                if qualifier in s.table_aliases:
                    return {(s.table_aliases[qualifier], name)}
                if qualifier in s.derived_aliases:
                    phys = self._resolve_in_derived(
                        s.derived_aliases[qualifier], name, schema_columns)
                    if phys:
                        return phys
                    raise _Unresolved(
                        f"qualifier '{qualifier}' has no column '{name}'")
                s = s.parent
            raise _Unresolved(f"qualifier '{qualifier}' is not a visible table/derived table/CTE")

        s = sc
        while s is not None:
            local = []
            for t in s.table_aliases.values():
                if t not in local:
                    local.append(t)
            if len(local) == 1:
                return {(local[0], name)}
            if schema_columns and len(local) > 1:
                matches = [t for t in local
                           if name.lower() in (self._schema_cols(t, schema_columns) or set())]
                if len(matches) == 1:
                    return {(matches[0], name)}
                if len(matches) > 1:
                    raise _Ambiguous(name, matches)
                # No match in this scope -> try this scope's derived outputs,
                # then go outward
            if s.derived_aliases:
                # Resolve per derived source: exactly one source hit
                # (including multi-side union ownership) -> return; several
                # hits -> ambiguous
                per_source = []
                for inner in s.derived_aliases.values():
                    r = self._resolve_in_derived(inner, name, schema_columns)
                    if r:
                        per_source.append(r)
                if len(per_source) == 1:
                    return per_source[0]
                if len(per_source) > 1:
                    tables_of = {t for r in per_source for t, _ in r}
                    raise _Ambiguous(name, tables_of)
            s = s.parent
        raise _Unresolved(f"unqualified column '{name}' cannot be uniquely assigned along the scope chain")

    # ------------------------------------------------------------------ main parse

    def _parse_with_sqlglot(self, sql: str, expected_table: str, dialect: str,
                            schema_columns: Optional[Dict[str, Set[str]]]) -> 'ParseResult':
        """Parse with sqlglot (scope-tree implementation)."""
        result = ParseResult(sql=sql, parse_method=f'sqlglot({dialect})')

        try:
            parsed = sqlglot.parse_one(sql, dialect=dialect)
        except Exception as e:  # noqa: BLE001
            result.success = False
            result.error = f"{type(e).__name__}: {str(e)[:300]}"
            return result

        try:
            self._scopes_cache = self._build_scopes(parsed)
            scopes = self._scopes_cache

            # 1. Physical table set (CTE names are not physical tables;
            # same-name tables merge under the simple name)
            physical = set()
            for sc in scopes.values():
                physical.update(sc.table_aliases.values())
            for name in sorted(physical):
                result.add_table(name)

            # 2. Extract column roles clause by clause, per scope
            for sc in scopes.values():
                select = sc.node
                # SELECT list (three-state stars + regular columns)
                for e in select.expressions:
                    stars = list(e.find_all(exp.Star))
                    if stars:
                        self._handle_star(stars, 'SELECT', sc, schema_columns, result)
                    for col in e.find_all(exp.Column):
                        if isinstance(col.this, exp.Star):
                            continue
                        self._add_column(col, 'SELECT', sc, schema_columns, result)

                for role, node in (('WHERE', select.args.get('where')),
                                   ('HAVING', select.args.get('having'))):
                    if node is None:
                        continue
                    for col in node.find_all(exp.Column):
                        self._add_column(col, role, sc, schema_columns, result)

                for join in (select.args.get('joins') or []):
                    on = join.args.get('on')
                    if on is not None:
                        for col in on.find_all(exp.Column):
                            self._add_column(col, 'JOIN', sc, schema_columns, result)

                group = select.args.get('group')
                if group is not None:
                    for e in group.expressions:
                        for col in e.find_all(exp.Column):
                            self._add_column(col, 'GROUP_BY', sc, schema_columns, result)

                order = select.args.get('order')
                if order is not None:
                    for ordered in order.expressions:
                        for col in ordered.find_all(exp.Column):
                            self._add_column(col, 'ORDER_BY', sc, schema_columns, result)

            # Statement-level ORDER BY (e.g. the outer side of a UNION, which
            # has no SELECT ancestor)
            for order in parsed.find_all(exp.Order):
                if order.find_ancestor(exp.Select) is not None:
                    continue
                root_scopes = list(scopes.values())
                if not root_scopes:
                    continue
                for ordered in order.expressions:
                    for col in ordered.find_all(exp.Column):
                        self._add_column(col, 'ORDER_BY', root_scopes[0],
                                         schema_columns, result)

            # Top-level projection set (the "clean projection" definition):
            # physical table columns reached by the output expressions of the
            # outermost SELECTs with no SELECT ancestor (star-expansion
            # targets included; both sides of a UNION count). Distinct from
            # field_roles, which mixes in all role columns of nested SELECTs.
            projection = set()
            for sel in list(parsed.find_all(exp.Select)):
                if sel.find_ancestor(exp.Select) is not None:
                    continue
                sc = scopes.get(id(sel))
                if sc is None:
                    continue
                for e in sel.expressions:
                    if list(e.find_all(exp.Star)):
                        agg = e.find_ancestor(exp.AggFunc)
                        if agg is not None:
                            continue  # a star inside an aggregate is not a projection column
                        node = e.find(exp.Star).parent
                        qualifier = (node.table or '').lower() \
                            if isinstance(node, exp.Column) else ''
                        if qualifier:
                            s2 = sc
                            while s2 is not None:
                                if qualifier in s2.table_aliases:
                                    projection |= {
                                        (s2.table_aliases[qualifier].lower(), c)
                                        for c in (self._schema_cols(
                                            s2.table_aliases[qualifier],
                                            schema_columns) or set())}
                                    break
                                if qualifier in s2.derived_aliases:
                                    for t in self._inner_physical_tables(
                                            s2.derived_aliases[qualifier]):
                                        projection |= {
                                            (t.lower(), c) for c in
                                            (self._schema_cols(t, schema_columns) or set())}
                                    break
                                s2 = s2.parent
                        else:
                            for t in self._visible_physical_tables(sc):
                                projection |= {
                                    (t.lower(), c) for c in
                                    (self._schema_cols(t, schema_columns) or set())}
                        continue
                    for col in e.find_all(exp.Column):
                        if isinstance(col.this, exp.Star):
                            continue
                        try:
                            for t, f in self._resolve_column(col, sc, schema_columns):
                                projection.add((t.lower(), f.lower()))
                        except (_Unresolved, _Ambiguous):
                            continue
            result.projection_fields = sorted(projection)

            # 3. JOIN table pairs (physical names; comma joins excluded)
            result.join_pairs = self._extract_join_pairs(parsed)

            # 4. WHERE predicate columns
            for sc in scopes.values():
                w = sc.node.args.get('where')
                if w is None:
                    continue
                for col in w.this.find_all(exp.Column):
                    try:
                        for t, f in self._resolve_column(col, sc, schema_columns):
                            result.predicates.append((t, f))
                    except (_Unresolved, _Ambiguous):
                        continue

            # Compatibility flags
            result.has_where = any(sc.node.args.get('where') for sc in scopes.values())
            result.has_join = bool(list(parsed.find_all(exp.Join)))
            result.has_group_by = any(sc.node.args.get('group') for sc in scopes.values())
            result.has_order_by = bool(list(parsed.find_all(exp.Order)))
            result.has_having = any(sc.node.args.get('having') for sc in scopes.values())

            result.success = True
            result.ast_parsed = True

        except Exception as e:  # noqa: BLE001
            result.success = False
            result.error = f"{type(e).__name__}: {str(e)[:300]}"

        return result

    # ------------------------------------------------------------------ star and column registration

    def _handle_star(self, stars, role: str, sc: '_Scope',
                     schema_columns, result: 'ParseResult'):
        """Three star states: SELECT * / t.* expand via the schema; a star as an aggregate argument is recorded, not expanded."""
        for star in stars:
            agg = star.find_ancestor(exp.AggFunc)
            if agg is not None:
                result.star_expansions.append(
                    {'form': 'aggregate_star', 'qualifier': '', 'tables': [],
                     'expanded_columns': 0,
                     'basis': f'star as the argument of aggregate {agg.sql_name()}(*); not expanded to physical columns'})
                continue

            qualifier = ''
            node = star.parent
            if isinstance(node, exp.Column):
                qualifier = (node.table or '').lower()

            if qualifier:
                target: Optional[List[str]] = None
                s = sc
                while s is not None and target is None:
                    if qualifier in s.table_aliases:
                        target = [s.table_aliases[qualifier]]
                    elif qualifier in s.derived_aliases:
                        target = self._inner_physical_tables(
                            s.derived_aliases[qualifier])
                    s = s.parent
                if not target:
                    result.unresolved.append(
                        {'item': f'{qualifier}.*', 'kind': 'star_qualifier',
                         'reason': f'qualifier {qualifier!r} cannot be resolved'})
                    continue
                self._expand_star_tables(target, role, schema_columns, result,
                                         form='table_star', qualifier=qualifier)
            else:
                tables = self._visible_physical_tables(sc)
                self._expand_star_tables(tables, role, schema_columns, result,
                                         form='select_star', qualifier='')

    def _inner_physical_tables(self, inner) -> List[str]:
        """All physical tables referenced inside a derived-table/CTE query (simple names, deduplicated)."""
        out, seen = [], set()
        if inner is None:
            return out
        nodes = [inner] if isinstance(inner, exp.Select) else list(inner.find_all(exp.Select))
        for node in nodes:
            sc = self._scopes_cache.get(id(node))
            if sc is None:
                continue
            for name in self._visible_physical_tables(sc):
                if name.lower() not in seen:
                    seen.add(name.lower())
                    out.append(name)
        return out

    def _expand_star_tables(self, tables: List[str], role: str, schema_columns,
                            result: 'ParseResult', form: str, qualifier: str):
        if not schema_columns:
            result.unresolved.append(
                {'item': f'{qualifier or "SELECT"}.*', 'kind': 'star_no_schema',
                 'reason': 'star expansion requires schema_columns, which was not provided'})
            return
        n = 0
        for t in tables:
            cols = self._schema_cols(t, schema_columns)
            if cols is None:
                result.unresolved.append(
                    {'item': f'{t}.*', 'kind': 'star_table_not_in_schema',
                     'reason': f'table {t} is not in the schema; star cannot be expanded'})
                continue
            for c in sorted(cols):
                result.add_field_role(t, c, role)
                n += 1
        result.star_expansions.append(
            {'form': form, 'qualifier': qualifier,
             'tables': sorted({t.lower() for t in tables}),
             'expanded_columns': n, 'basis': 'expanded to all schema_columns'})

    def _add_column(self, col: 'exp.Column', role: str, sc: '_Scope',
                    schema_columns, result: 'ParseResult'):
        """Resolve and register one column node (unresolved / ambiguous recorded explicitly).

        Columns are always resolved in the scope of their **nearest enclosing
        SELECT** (an outer clause's find_all descends into subqueries, so the
        outer scope must not duplicate or mis-attribute them); the clause
        scope is only a fallback.
        """
        name = col.name
        if not name:
            return
        scope = self._scope_for_node(col) or sc
        try:
            resolved = self._resolve_column(col, scope, schema_columns)
        except _Ambiguous as amb:
            result.ambiguous.append(
                {'column': name, 'candidates': sorted({c.lower() for c in amb.candidates}),
                 'role': role})
            result.warnings.append(
                f"ambiguous column '{name}' across {sorted({c.lower() for c in amb.candidates})}")
            return
        except _Unresolved as un:
            result.unresolved.append(
                {'item': f"{(col.table + '.') if col.table else ''}{name}",
                 'kind': 'unresolved_column', 'role': role, 'reason': str(un)})
            result.warnings.append(
                f"cannot resolve column '{name}' (qualifier='{col.table}'): {un}")
            return
        for table, fld in resolved:
            if schema_columns is not None:
                cols = self._schema_cols(table, schema_columns)
                if cols is not None and fld.lower() not in cols:
                    result.unresolved.append(
                        {'item': f'{table}.{fld}', 'kind': 'not_in_schema',
                         'role': role,
                         'reason': f'column not in the schema of table {table} (possibly an output alias / derived column)'})
                    result.warnings.append(f"column '{table}.{fld}' not in schema")
                    continue
            result.add_field_role(table, fld, role)

    def _extract_join_pairs(self, parsed) -> List[Tuple[str, str]]:
        """Extract the table pairs joined by JOIN clauses (physical names; comma joins excluded)."""
        pairs = []
        try:
            for join in parsed.find_all(exp.Join):
                names: List[str] = []
                for t in join.find_all(exp.Table):
                    if t.name and t.name not in names:
                        names.append(t.name)
                if len(names) >= 2:
                    for other in names[1:]:
                        pair = (names[0].lower(), other.lower())
                        if pair not in pairs:
                            pairs.append(pair)
        except Exception:  # noqa: BLE001
            pass
        return pairs

    # ------------------------------------------------------------------ regex fallback

    def _parse_with_regex(self, sql: str, expected_table: str = None) -> 'ParseResult':
        """Parse with regular expressions (fallback; not a complete AST parse, so ast_parsed=False)."""
        result = ParseResult(sql=sql, parse_method='regex')

        try:
            sql_upper = sql.upper()

            from_match = re.search(r'FROM\s+([`\w]+)', sql_upper)
            if from_match:
                result.add_table(from_match.group(1).strip('`').lower())

            join_match = re.search(r'JOIN\s+([`\w]+)', sql_upper)
            if join_match:
                result.add_table(join_match.group(1).strip('`').lower())

            if 'WHERE' in sql_upper:
                result.has_where = True
            if 'JOIN' in sql_upper:
                result.has_join = True
            if 'GROUP BY' in sql_upper:
                result.has_group_by = True
            if 'ORDER BY' in sql_upper:
                result.has_order_by = True
            if 'HAVING' in sql_upper:
                result.has_having = True

            result.success = True

        except Exception as e:  # noqa: BLE001
            result.success = False
            result.error = str(e)

        return result

    def get_role_statistics(self, sql_list: List[str]) -> Dict[str, int]:
        """
        Count how often each role appears across a list of SQL statements.

        Args:
            sql_list: list of SQL statements

        Returns:
            Dict of role -> frequency.
        """
        stats = defaultdict(int)

        for sql in sql_list:
            result = self.parse(sql)
            if result.success:
                for table, fields in result.field_roles.items():
                    for field, info in fields.items():
                        for role in info.roles:
                            stats[role] += 1

        return dict(stats)


@dataclass
class ParseResult:
    """Result of parsing one SQL statement."""
    sql: str
    success: bool = False
    error: str = ""
    parse_method: str = "unknown"

    # Successful-AST flag (independent of success: the regex fallback has
    # success=True but ast_parsed=False)
    ast_parsed: bool = False

    # Table name set (lowercase)
    tables: Set[str] = field(default_factory=set)

    # Column role info {table: {field: FieldRoleInfo}} (field names keep
    # their original case)
    field_roles: Dict[str, Dict[str, FieldRoleInfo]] = field(default_factory=lambda: defaultdict(dict))

    # JOIN table pairs (real table names, lowercase)
    join_pairs: List[Tuple[str, str]] = field(default_factory=list)

    # WHERE predicate columns [(table, column), ...]
    predicates: List[Tuple[str, str]] = field(default_factory=list)

    # Parse warnings (human-readable summary; structured detail lives in
    # unresolved/ambiguous/star_expansions)
    warnings: List[str] = field(default_factory=list)

    # Unresolved items (gold construction must handle each one; never treat
    # them as covered)
    unresolved: List[Dict] = field(default_factory=list)

    # Columns ambiguous across tables (no unique owner; must not be silently
    # written into gold)
    ambiguous: List[Dict] = field(default_factory=list)

    # Star-handling records (form: select_star/table_star/aggregate_star)
    star_expansions: List[Dict] = field(default_factory=list)

    # Top-level projection columns [(table, column), ...] (the
    # clean-projection definition: only outermost SELECT outputs, excluding
    # columns of nested subquery SELECTs)
    projection_fields: List[Tuple[str, str]] = field(default_factory=list)

    # Simplified flags (used by the regex parsing fallback)
    has_where: bool = False
    has_join: bool = False
    has_group_by: bool = False
    has_order_by: bool = False
    has_having: bool = False

    def add_table(self, table: str):
        """Add a table name (stored lowercase)."""
        if table:
            self.tables.add(table.lower())

    def add_field_role(self, table: str, field: str, role: str):
        """Register a role for a table column."""
        table = table.lower()
        if not field:
            return

        if field not in self.field_roles[table]:
            self.field_roles[table][field] = FieldRoleInfo(table=table, field=field)

        self.field_roles[table][field].add_role(role)

    def find_field_info(self, table: str, field: str) -> Optional[FieldRoleInfo]:
        """Look up column role information case-insensitively."""
        table = table.lower() if table else ''
        field_lower = field.lower() if field else ''
        table_dict = self.field_roles.get(table, {})
        if field_lower in table_dict:
            return table_dict[field_lower]
        for f, info in table_dict.items():
            if f.lower() == field_lower:
                return info
        return None

    def get_field_weight(self, table: str, field: str,
                         role_weights: Dict[str, float] = None) -> float:
        """
        Return the weight of a table column.

        Args:
            table: table name
            field: column name
            role_weights: role weight dict

        Returns:
            The column weight.
        """
        role_weights = role_weights or {
            'SELECT': 1.2,
            'WHERE': 2.0,
            'JOIN': 1.8,
            'GROUP_BY': 1.5,
            'ORDER_BY': 1.3,
            'HAVING': 1.4,
        }

        info = self.find_field_info(table, field)
        if info is not None:
            return info.get_max_weight(role_weights)

        # Fallback heuristic for regex-parsed results
        weight = 1.0
        if self.has_where:
            weight = max(weight, 2.0)
        if self.has_join:
            weight = max(weight, 1.8)
        if self.has_group_by:
            weight = max(weight, 1.5)
        if self.has_order_by:
            weight = max(weight, 1.3)

        return weight

    def get_all_fields(self) -> Set[Tuple[str, str]]:
        """Return all (table, field) pairs; table lowercase, field keeps original case."""
        fields = set()
        for table, field_dict in self.field_roles.items():
            for field in field_dict:
                fields.add((table, field))
        return fields

    def get_gold_columns(self) -> List[str]:
        """Return gold columns as 'table.column' strings (for multi-table evaluation)."""
        return sorted(f"{t}.{f}" for t, f in self.get_all_fields())

    def get_gold_tables(self) -> List[str]:
        """Return gold tables (lowercase)."""
        return sorted(self.tables)


def demo():
    """Demonstrate SQL role parsing."""
    parser = SQLRoleParser(dialect='sqlite')

    schema = {
        'schools': {'CDSCode', 'Street', 'City'},
        'satscores': {'cds', 'AvgScrRead'},
        'frpm': {'Free Meal Count (K-12)', 'Enrollment (K-12)', 'County Name'},
    }

    # Sample SQL statements covering typical cases
    test_sqls = [
        # single table + backquoted column names
        "SELECT `Free Meal Count (K-12)` FROM frpm WHERE `County Name` = 'Alameda'",
        # JOIN + aliases
        "SELECT T2.Street, T2.City FROM schools AS T2 JOIN satscores AS T1 ON T2.CDSCode = T1.cds",
        # aggregation + ORDER BY + LIMIT
        ("SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm "
         "WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / "
         "`Enrollment (K-12)`) DESC LIMIT 1"),
        # subquery
        "SELECT name FROM schools WHERE CDSCode IN (SELECT cds FROM satscores WHERE AvgScrRead > 500)",
    ]

    for sql in test_sqls:
        print(f"\nSQL: {sql[:80]}...")
        result = parser.parse(sql, dialect='sqlite', schema_columns=schema)

        if result.success:
            print(f"  parse method: {result.parse_method}")
            print(f"  gold tables: {result.get_gold_tables()}")
            print(f"  gold columns: {result.get_gold_columns()}")
            if result.warnings:
                print(f"  warnings: {result.warnings}")
        else:
            print(f"  parse failed: {result.error}")


if __name__ == "__main__":
    demo()
