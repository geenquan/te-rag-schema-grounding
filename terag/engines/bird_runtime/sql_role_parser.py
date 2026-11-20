"""SQL role parser.

Parses SQL with sqlglot ASTs to accurately identify each field's role in
the statement.

Supports:
- Multiple dialects (mysql default; ODPS/Hive-style comments and functions
  automatically fall back to the hive dialect; BIRD configurable as sqlite)
- Multi-table SQL (JOIN / subqueries / derived tables / CTEs); gold tables
  are all physical tables
- Scoped alias resolution (T1/T2, derived-table aliases, CTE names;
  subquery scopes stay independent, no global cross-contamination)
- Unqualified column resolution (unique visible table / unique schema
  match; multi-table ambiguity is recorded explicitly, never silently
  resolved to the first candidate)
- Three star states: SELECT * and t.* expand against the schema; an
  aggregate argument * is only recorded, not expanded
- Parse-completeness flag kept separate: ast_parsed != all fields
  resolvable; unresolved / ambiguous / star_expansions are recorded item
  by item so a "success with zero fields" never hides a failure
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
    """A column shares its name across several visible tables."""

    def __init__(self, column: str, candidates):
        super().__init__(column)
        self.column = column
        self.candidates = list(candidates)


@dataclass
class FieldRoleInfo:
    """Field role information."""
    table: str
    field: str
    roles: Set[str] = field(default_factory=set)  # SELECT, WHERE, JOIN, GROUP_BY, ORDER_BY, HAVING
    role_weights: Dict[str, float] = field(default_factory=dict)

    def add_role(self, role: str, weight: float = 1.0):
        """Add a role."""
        self.roles.add(role)
        self.role_weights[role] = max(self.role_weights.get(role, 0), weight)

    def get_max_weight(self, role_weights: Dict[str, float]) -> float:
        """Get the maximum weight."""
        if not self.roles:
            return 1.0
        return max(role_weights.get(r, 1.0) for r in self.roles)


class SQLRoleParser:
    """SQL role parser.

    Parses SQL with sqlglot, resolving each field's physical attribution
    and role per SELECT scope.

    Supported roles:
    - SELECT / WHERE / JOIN / GROUP_BY / ORDER_BY / HAVING / FROM

    Usage:
        parser = SQLRoleParser(dialect='sqlite')
        result = parser.parse(sql, schema_columns={'frpm': {'County Name', ...}})
        print(result.tables)        # all physical tables
        print(result.field_roles)   # {table_name: {field_name: FieldRoleInfo}}
        print(result.unresolved)    # unresolved items (gold builders must check)
        print(result.ambiguous)     # multi-table ambiguous columns
    """

    def __init__(self, default_role_weights: Dict[str, float] = None,
                 dialect: str = 'mysql', mode: str = 'ast'):
        """Initialize the parser.

        Args:
            default_role_weights: default role weights.
            dialect: SQL dialect (PowerGrid defaults to mysql; ODPS
                comments/functions auto-fall back to hive).
            mode: 'ast' (default; the P2-A validated AST parse) or 'regex'
                (historical regex fallback, diagnostics only; not a
                complete parse).
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
            raise ValueError(f"Unknown parse mode: {mode}")
        self.mode = mode
        self._scopes_cache: Dict[int, 'SQLRoleParser._Scope'] = {}

    def parse(self, sql: str, expected_table: str = None,
              dialect: str = None,
              schema_columns: Optional[Dict[str, Set[str]]] = None) -> 'ParseResult':
        """Parse a SQL statement.

        Args:
            sql: the SQL statement.
            expected_table: expected table name (parameter kept; scoped
                resolution no longer uses it to guess attribution).
            dialect: SQL dialect (overrides the default).
            schema_columns: physical schema {table: {column, ...}} used for
                unqualified-column disambiguation, derived-table/CTE output
                matching and star expansion. Complete gold construction must
                provide it; without it multi-table unqualified columns are
                recorded as unresolved rather than guessed.

        Returns:
            ParseResult (ast_parsed means the AST parse succeeded;
            field-level completeness is in unresolved / ambiguous /
            star_expansions)
        """
        if self.mode == 'regex' or not SQLGLOT_AVAILABLE:
            result = self._parse_with_regex(sql, expected_table)
            result.ast_parsed = False
            reason = ('explicit regex mode (diagnostics)' if self.mode == 'regex'
                      else 'sqlglot-not-installed')
            result.unresolved.append({'item': reason,
                                      'kind': 'parser_missing',
                                      'reason': 'regex fallback, not a complete AST parse'})
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
        """Scope of one SELECT: visible physical table aliases, derived-table
        aliases and CTEs."""

        def __init__(self, node, parent):
            self.node = node      # exp.Select
            self.parent = parent  # outer _Scope (correlated subqueries see outer tables)
            self.table_aliases: Dict[str, str] = {}       # alias/table(lower) -> physical table name
            self.derived_aliases: Dict[str, object] = {}  # derived/CTE alias -> inner query

    def _build_scopes(self, parsed) -> Dict[int, '_Scope']:
        """Build a scope per exp.Select (outer relation + FROM/JOIN sources +
        visible CTEs)."""
        selects = list(parsed.find_all(exp.Select))
        parent_select = {}
        for s in selects:
            p = s.parent
            while p is not None and not isinstance(p, exp.Select):
                p = p.parent
            parent_select[id(s)] = p

        # CTE definitions (WITH clauses): name -> inner query
        cte_defs: Dict[str, object] = {}
        for cte in parsed.find_all(exp.CTE):
            cte_defs[cte.alias_or_name.lower()] = cte.this

        scopes: Dict[int, 'SQLRoleParser._Scope'] = {}
        for s in selects:
            parent_scope = scopes.get(id(parent_select[id(s)])) \
                if parent_select.get(id(s)) is not None else None
            sc = self._Scope(s, parent_scope)

            sources = []
            # sqlglot >=30 uses 'from_' (avoiding the Python keyword); older
            # versions use 'from'
            frm = s.args.get('from_') or s.args.get('from')
            if frm is not None:
                # All direct child sources of From (normally only 'this';
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
                # FROM-tree preorder walk (stops at Select boundaries):
                # aliased Subquery/Table under parenthesized JOINs or odd
                # wrappers stays visible in this scope
                self._collect_nested_sources(src, sc, cte_defs, depth=0)

            scopes[id(s)] = sc
        return scopes

    def _collect_nested_sources(self, node, sc: '_Scope',
                                cte_defs: Dict[str, object], depth: int):
        """Walk the FROM source tree registering aliased Subquery/Table nodes;
        does not enter inner Selects."""
        if node is None or depth > 12:
            return
        if isinstance(node, exp.Select):
            return  # Select boundary: the inner scope handles itself
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
        """Register one FROM/JOIN source (physical table / derived table /
        CTE reference / parenthesized JOIN)."""
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
            # Recursively strip anonymous parenthesized wrappers; the real
            # alias may sit on an inner Subquery
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
                # Parenthesized JOIN wrapper: FROM ( (...) a JOIN (...) b )
                # -- transparently expand
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
            # Set operations like Union act as derived sources directly
            alias = (getattr(src, 'alias', '') or '')
            key = alias.lower() if alias else f'__anon_derived_{len(sc.derived_aliases)}'
            sc.derived_aliases[key] = src

    def _visible_physical_tables(self, sc: '_Scope') -> List[str]:
        """All physical tables visible along the scope chain (deduplicated,
        appearance order kept)."""
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
        """Scope of the closest SELECT owning the node."""
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
        """Locate the physical (table, column) set matching column_name in a
        derived-table/CTE output."""
        if depth > 8:
            return None
        if not isinstance(inner, exp.Select):
            # Set operations like UNION: search both sides
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
            # Explicit output column missed but the inner query has a star:
            # search among the star-expanded physical columns
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
        """Resolve a column node -> physical (table, column) set.

        Qualified column: walk the scope chain for the alias/derived
        table/CTE (inner naturally shadows outer).
        Unqualified column (local-first, matching SQL name resolution):
        single local physical table -> attribute to it; unique local schema
        match -> attribute to it; local ambiguity -> record ambiguity; no
        local match -> try this scope's derived outputs; then repeat in the
        outer scope.

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
                        f"Column '{name}' not found in the output of qualifier '{qualifier}'")
                s = s.parent
            raise _Unresolved(f"Qualifier '{qualifier}' is not a visible table/derived table/CTE")

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
                # No local match -> try this scope's derived outputs, then
                # the outer scope
            if s.derived_aliases:
                # Resolve per derived source: a single hit (including a
                # multi-side union with multiple attributions) -> return;
                # multiple hits -> ambiguity
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
        raise _Unresolved(f"Unqualified column '{name}' cannot be uniquely attributed along the scope chain")

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
            # same-name tables merge by simple name)
            physical = set()
            for sc in scopes.values():
                physical.update(sc.table_aliases.values())
            for name in sorted(physical):
                result.add_table(name)

            # 2. Extract field roles per scope and clause
            for sc in scopes.values():
                select = sc.node
                # SELECT list (three star states + normal columns)
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

            # Statement-level ORDER BY (e.g. a UNION outer without a SELECT
            # ancestor)
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

            # Top-level projection set (clean_projection semantics): the
            # physical table columns reached by the output expressions of
            # the outermost SELECTs without a SELECT ancestor (including
            # star-expansion targets; both UNION sides counted). Distinct
            # from field_roles, which mixes in all role columns of nested
            # subquery SELECTs.
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
                            continue  # an aggregate star is not a projection column
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

            # 3. JOIN pairs (physical table names; comma joins excluded)
            result.join_pairs = self._extract_join_pairs(parsed)

            # 4. WHERE predicate fields
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
        """Three star states: SELECT * / t.* expand against the schema; an
        aggregate argument * is only recorded, not expanded."""
        for star in stars:
            agg = star.find_ancestor(exp.AggFunc)
            if agg is not None:
                result.star_expansions.append(
                    {'form': 'aggregate_star', 'qualifier': '', 'tables': [],
                     'expanded_columns': 0,
                     'basis': f'aggregate {agg.sql_name()}(*) argument star, not expanded to physical columns'})
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
        """All physical tables referenced inside a derived table/CTE (simple
        names, deduplicated)."""
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
                 'reason': 'star expansion requires schema_columns, not provided'})
            return
        n = 0
        for t in tables:
            cols = self._schema_cols(t, schema_columns)
            if cols is None:
                result.unresolved.append(
                    {'item': f'{t}.*', 'kind': 'star_table_not_in_schema',
                     'reason': f'table {t} not in the schema, star cannot expand'})
                continue
            for c in sorted(cols):
                result.add_field_role(t, c, role)
                n += 1
        result.star_expansions.append(
            {'form': form, 'qualifier': qualifier,
             'tables': sorted({t.lower() for t in tables}),
             'expanded_columns': n, 'basis': 'full schema_columns expansion'})

    def _add_column(self, col: 'exp.Column', role: str, sc: '_Scope',
                    schema_columns, result: 'ParseResult'):
        """Resolve and register a column node (unresolved / ambiguous
        recorded explicitly).

        Columns are always resolved against their NEAREST SELECT scope (an
        outer clause's find_all descends into subqueries, so the outer
        scope must not re-attribute or double-count); the clause scope is
        the fallback.
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
                         'reason': f'column not in the schema of table {table} (likely an output alias/derived column)'})
                    result.warnings.append(f"column '{table}.{fld}' not in schema")
                    continue
            result.add_field_role(table, fld, role)

    def _extract_join_pairs(self, parsed) -> List[Tuple[str, str]]:
        """Extract JOIN pairs (physical table names; comma joins excluded)."""
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
        """Parse with regular expressions (fallback; not a complete AST
        parse, ast_parsed=False)."""
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
        """Role frequency statistics over SQL statements.

        Args:
            sql_list: list of SQL statements.

        Returns:
            Role frequency dict.
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
    """Parse result."""
    sql: str
    success: bool = False
    error: str = ""
    parse_method: str = "unknown"

    # AST parse success flag (independent of success: the regex fallback
    # has success=True but ast_parsed=False)
    ast_parsed: bool = False

    # Table name set (lowercase)
    tables: Set[str] = field(default_factory=set)

    # Field role info {table: {field: FieldRoleInfo}} (fields keep original case)
    field_roles: Dict[str, Dict[str, FieldRoleInfo]] = field(default_factory=lambda: defaultdict(dict))

    # JOIN pairs (physical table names, lowercase)
    join_pairs: List[Tuple[str, str]] = field(default_factory=list)

    # WHERE predicate fields [(table, column), ...]
    predicates: List[Tuple[str, str]] = field(default_factory=list)

    # Parse warnings (human-readable summaries; structured detail is in
    # unresolved/ambiguous/star_expansions)
    warnings: List[str] = field(default_factory=list)

    # Unresolved items (gold builders must handle each; never count them covered)
    unresolved: List[Dict] = field(default_factory=list)

    # Multi-table ambiguous columns (no unique attribution; never silently
    # fill them into gold)
    ambiguous: List[Dict] = field(default_factory=list)

    # Star handling records (form: select_star/table_star/aggregate_star)
    star_expansions: List[Dict] = field(default_factory=list)

    # Top-level projection physical columns [(table, column), ...]
    # (clean_projection semantics; only the outermost SELECT outputs, no
    # inner subquery SELECT columns)
    projection_fields: List[Tuple[str, str]] = field(default_factory=list)

    # Simplified flags (for the regex fallback)
    has_where: bool = False
    has_join: bool = False
    has_group_by: bool = False
    has_order_by: bool = False
    has_having: bool = False

    def add_table(self, table: str):
        """Add a table."""
        if table:
            self.tables.add(table.lower())

    def add_field_role(self, table: str, field: str, role: str):
        """Add a field role."""
        table = table.lower()
        if not field:
            return

        if field not in self.field_roles[table]:
            self.field_roles[table][field] = FieldRoleInfo(table=table, field=field)

        self.field_roles[table][field].add_role(role)

    def find_field_info(self, table: str, field: str) -> Optional[FieldRoleInfo]:
        """Case-insensitive field role lookup."""
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
        """Get a field weight.

        Args:
            table: table name.
            field: field name.
            role_weights: role weight dict.

        Returns:
            The field weight.
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

        # Regex-parse fallback logic
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
        """All fields (table, field); table lowercase, field original case."""
        fields = set()
        for table, field_dict in self.field_roles.items():
            for field in field_dict:
                fields.add((table, field))
        return fields

    def get_gold_columns(self) -> List[str]:
        """Gold columns (table.column, for multi-table evaluation)."""
        return sorted(f"{t}.{f}" for t, f in self.get_all_fields())

    def get_gold_tables(self) -> List[str]:
        """Gold tables (lowercase)."""
        return sorted(self.tables)


def demo():
    """SQL role parser demo."""
    parser = SQLRoleParser(dialect='sqlite')

    schema = {
        'schools': {'CDSCode', 'Street', 'City'},
        'satscores': {'cds', 'AvgScrRead'},
        'frpm': {'Free Meal Count (K-12)', 'Enrollment (K-12)', 'County Name'},
    }

    # Test SQL (typical scenarios)
    test_sqls = [
        # Single table + backticked column names
        "SELECT `Free Meal Count (K-12)` FROM frpm WHERE `County Name` = 'Alameda'",
        # JOIN + aliases
        "SELECT T2.Street, T2.City FROM schools AS T2 JOIN satscores AS T1 ON T2.CDSCode = T1.cds",
        # Aggregation + ORDER BY + LIMIT
        ("SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm "
         "WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / "
         "`Enrollment (K-12)`) DESC LIMIT 1"),
        # Subquery
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
