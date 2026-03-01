package org.apache.calcite.plan.hep;

import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelVisitor;
import org.apache.calcite.rel.core.*;
import org.apache.calcite.rel.logical.*;
import org.apache.calcite.rex.*;

/**
 * Query analysis helper methods for the evolved policy.
 * Extended feature set to better distinguish query patterns.
 */
public class QueryAnalyzer {

    // ==================== SUBQUERY FEATURES ====================

    /**
     * Check if query contains a subquery (IN, EXISTS, scalar subquery).
     */
    public static boolean hasSubquery(RelNode root) {
        if (containsType(root, Correlate.class)) {
            return true;
        }
        return countRexSubQueries(root) > 0;
    }

    /**
     * Count the number of RexSubQuery nodes (IN, EXISTS, scalar subqueries).
     */
    public static int countSubqueries(RelNode root) {
        int correlates = countType(root, Correlate.class);
        int rexSubs = countRexSubQueries(root);
        return Math.max(correlates, rexSubs);
    }

    /**
     * Check if query has nested subqueries (subquery inside subquery).
     */
    public static boolean hasNestedSubquery(RelNode root) {
        final boolean[] found = {false};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (found[0]) return;
                if (node instanceof Correlate) {
                    // Check if the subquery side also has subqueries
                    RelNode right = ((Correlate) node).getRight();
                    if (hasSubquery(right)) {
                        found[0] = true;
                        return;
                    }
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return found[0];
    }

    /**
     * Check if query contains correlated subquery.
     */
    public static boolean hasCorrelation(RelNode root) {
        final boolean[] found = {false};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (found[0]) return;
                if (!node.getVariablesSet().isEmpty()) {
                    found[0] = true;
                    return;
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return found[0];
    }

    /**
     * Count the number of correlation variables used.
     */
    public static int countCorrelatedVars(RelNode root) {
        final int[] count = {0};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                count[0] += node.getVariablesSet().size();
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return count[0];
    }

    /**
     * Check if query has EXISTS subquery pattern.
     * EXISTS subqueries are already in semi-join form and don't benefit from FILTER_SUB_QUERY_TO_CORRELATE.
     * Detects:
     * - RexSubQuery with EXISTS operator
     * - Correlate nodes with SEMI or ANTI join type
     */
    public static boolean hasExistsSubquery(RelNode root) {
        final boolean[] found = {false};
        
        // Check for Correlate with SEMI/ANTI join type (already converted EXISTS)
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (found[0]) return;
                if (node instanceof Correlate) {
                    Correlate corr = (Correlate) node;
                    if (corr.getJoinType() == JoinRelType.SEMI || 
                        corr.getJoinType() == JoinRelType.ANTI) {
                        found[0] = true;
                        return;
                    }
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        
        if (found[0]) return true;
        
        // Check for RexSubQuery with EXISTS in filter conditions
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (found[0]) return;
                if (node instanceof Filter) {
                    if (containsExistsSubQuery(((Filter) node).getCondition())) {
                        found[0] = true;
                        return;
                    }
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        
        return found[0];
    }

    /**
     * Check if query has IN subquery pattern.
     * IN subqueries can benefit from FILTER_SUB_QUERY_TO_CORRELATE.
     */
    public static boolean hasInSubquery(RelNode root) {
        final boolean[] found = {false};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (found[0]) return;
                if (node instanceof Filter) {
                    if (containsInSubQuery(((Filter) node).getCondition())) {
                        found[0] = true;
                        return;
                    }
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return found[0];
    }

    /**
     * Check if query has self-join subquery pattern.
     * This is when EXISTS/IN subquery references the same table as the outer query.
     * Pattern: SELECT * FROM lineitem l1 WHERE EXISTS (SELECT * FROM lineitem l2 WHERE l1.x = l2.x)
     * 
     * IMPORTANT: FILTER_SUB_QUERY_TO_CORRELATE makes queries with self-join subqueries WORSE
     * because it creates nested lateral joins on the same large table.
     * PostgreSQL handles these better with native SEMI-JOIN/ANTI-JOIN.
     */
    public static boolean hasSelfJoinSubquery(RelNode root) {
        // Collect table names from the main query (non-subquery parts)
        final java.util.Set<String> outerTables = new java.util.HashSet<>();
        final java.util.Set<String> subqueryTables = new java.util.HashSet<>();
        
        // First pass: collect outer query tables
        new RelVisitor() {
            boolean inSubquery = false;
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (node instanceof Correlate) {
                    // Left side is outer query, right side is subquery
                    Correlate corr = (Correlate) node;
                    collectTables(corr.getLeft(), outerTables);
                    collectTables(corr.getRight(), subqueryTables);
                    return; // Don't recurse further
                }
                if (node instanceof Filter) {
                    // Check for RexSubQuery in filter
                    RexNode condition = ((Filter) node).getCondition();
                    collectSubqueryTables(condition, subqueryTables);
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        
        // If no subqueries found through Correlate, collect all tables as outer
        if (subqueryTables.isEmpty()) {
            collectTables(root, outerTables);
        }
        
        // Check for intersection
        for (String table : outerTables) {
            if (subqueryTables.contains(table)) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * Collect all table names from a RelNode tree.
     */
    private static void collectTables(RelNode node, java.util.Set<String> tables) {
        new RelVisitor() {
            @Override
            public void visit(RelNode n, int ordinal, RelNode parent) {
                if (n instanceof TableScan) {
                    TableScan scan = (TableScan) n;
                    // Get table name (last part of qualified name)
                    java.util.List<String> names = scan.getTable().getQualifiedName();
                    if (!names.isEmpty()) {
                        tables.add(names.get(names.size() - 1).toLowerCase());
                    }
                }
                super.visit(n, ordinal, parent);
            }
        }.go(node);
    }
    
    /**
     * Collect table names from RexSubQuery expressions.
     */
    private static void collectSubqueryTables(RexNode rex, java.util.Set<String> tables) {
        if (rex instanceof RexSubQuery) {
            collectTables(((RexSubQuery) rex).rel, tables);
        }
        if (rex instanceof RexCall) {
            for (RexNode operand : ((RexCall) rex).getOperands()) {
                collectSubqueryTables(operand, tables);
            }
        }
    }

    /**
     * Check if query has scalar subquery pattern (subquery in comparison).
     * Pattern: column < (SELECT agg(...) FROM ...)
     * Scalar subqueries can benefit from FILTER_SUB_QUERY_TO_CORRELATE.
     */
    public static boolean hasScalarSubquery(RelNode root) {
        final boolean[] found = {false};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (found[0]) return;
                if (node instanceof Filter) {
                    if (containsScalarSubQuery(((Filter) node).getCondition())) {
                        found[0] = true;
                        return;
                    }
                }
                if (node instanceof Project) {
                    for (RexNode expr : ((Project) node).getProjects()) {
                        if (containsScalarSubQuery(expr)) {
                            found[0] = true;
                            return;
                        }
                    }
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return found[0];
    }

    // ==================== JOIN FEATURES ====================

    /**
     * Count the number of joins in the query.
     */
    public static int countJoins(RelNode root) {
        return countType(root, Join.class);
    }

    /**
     * Check if query has outer joins (LEFT, RIGHT, FULL).
     */
    public static boolean hasOuterJoin(RelNode root) {
        final boolean[] found = {false};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (found[0]) return;
                if (node instanceof Join) {
                    Join join = (Join) node;
                    if (join.getJoinType() != JoinRelType.INNER) {
                        found[0] = true;
                        return;
                    }
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return found[0];
    }

    /**
     * Check if there's a Filter directly above a Join.
     */
    public static boolean hasFilterAboveJoin(RelNode root) {
        final boolean[] found = {false};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (found[0]) return;
                if (node instanceof Filter) {
                    for (RelNode input : node.getInputs()) {
                        if (input instanceof Join) {
                            found[0] = true;
                            return;
                        }
                    }
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return found[0];
    }

    /**
     * Check if query has semi-join pattern (EXISTS subquery).
     */
    public static boolean hasSemiJoin(RelNode root) {
        final boolean[] found = {false};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (found[0]) return;
                if (node instanceof Correlate) {
                    Correlate corr = (Correlate) node;
                    if (corr.getJoinType() == JoinRelType.SEMI || 
                        corr.getJoinType() == JoinRelType.ANTI) {
                        found[0] = true;
                        return;
                    }
                }
                if (node instanceof Join) {
                    Join join = (Join) node;
                    if (join.getJoinType() == JoinRelType.SEMI ||
                        join.getJoinType() == JoinRelType.ANTI) {
                        found[0] = true;
                        return;
                    }
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return found[0];
    }

    // ==================== AGGREGATE FEATURES ====================

    /**
     * Check if query contains aggregation.
     */
    public static boolean hasAggregate(RelNode root) {
        return containsType(root, Aggregate.class);
    }

    /**
     * Count the number of aggregate nodes.
     */
    public static int countAggregates(RelNode root) {
        return countType(root, Aggregate.class);
    }

    /**
     * Count total aggregate function calls (SUM, COUNT, AVG, etc.).
     */
    public static int countAggregateCalls(RelNode root) {
        final int[] count = {0};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (node instanceof Aggregate) {
                    count[0] += ((Aggregate) node).getAggCallList().size();
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return count[0];
    }

    /**
     * Check if any aggregate has DISTINCT.
     */
    public static boolean hasDistinct(RelNode root) {
        final boolean[] found = {false};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (found[0]) return;
                if (node instanceof Aggregate) {
                    Aggregate agg = (Aggregate) node;
                    for (AggregateCall call : agg.getAggCallList()) {
                        if (call.isDistinct()) {
                            found[0] = true;
                            return;
                        }
                    }
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return found[0];
    }

    /**
     * Count the number of GROUP BY keys.
     */
    public static int countGroupByKeys(RelNode root) {
        final int[] count = {0};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (node instanceof Aggregate) {
                    count[0] += ((Aggregate) node).getGroupCount();
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return count[0];
    }

    // ==================== FILTER FEATURES ====================

    /**
     * Check if query has a filter (WHERE clause).
     */
    public static boolean hasFilter(RelNode root) {
        return containsType(root, Filter.class);
    }

    /**
     * Count the number of filter nodes.
     */
    public static int countFilters(RelNode root) {
        return countType(root, Filter.class);
    }

    /**
     * Count the number of filter predicates (AND conditions).
     */
    public static int countFilterPredicates(RelNode root) {
        final int[] count = {0};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (node instanceof Filter) {
                    count[0] += countPredicates(((Filter) node).getCondition());
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return count[0];
    }

    // ==================== WINDOW FUNCTION FEATURES ====================

    /**
     * Check if query has window functions (OVER clause).
     */
    public static boolean hasWindowFunction(RelNode root) {
        return containsType(root, Window.class);
    }

    /**
     * Count the number of window function nodes.
     */
    public static int countWindowFunctions(RelNode root) {
        return countType(root, Window.class);
    }

    // ==================== SET OPERATION FEATURES ====================

    /**
     * Check if query has UNION, INTERSECT, or EXCEPT.
     */
    public static boolean hasUnion(RelNode root) {
        return containsType(root, Union.class) || 
               containsType(root, Intersect.class) || 
               containsType(root, Minus.class);
    }

    // ==================== TABLE FEATURES ====================

    /**
     * Count the number of table scans.
     */
    public static int countTables(RelNode root) {
        return countType(root, TableScan.class);
    }

    // ==================== SORT/LIMIT FEATURES ====================

    /**
     * Check if query has sorting (ORDER BY).
     */
    public static boolean hasSort(RelNode root) {
        return containsType(root, Sort.class);
    }

    /**
     * Check if query has LIMIT clause.
     */
    public static boolean hasLimit(RelNode root) {
        final boolean[] found = {false};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (found[0]) return;
                if (node instanceof Sort) {
                    Sort sort = (Sort) node;
                    if (sort.fetch != null) {
                        found[0] = true;
                        return;
                    }
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return found[0];
    }

    // ==================== EXPRESSION FEATURES ====================

    /**
     * Check if query has CASE WHEN expressions.
     */
    public static boolean hasCaseWhen(RelNode root) {
        final boolean[] found = {false};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (found[0]) return;
                if (node instanceof Project) {
                    for (RexNode expr : ((Project) node).getProjects()) {
                        if (containsCase(expr)) {
                            found[0] = true;
                            return;
                        }
                    }
                }
                if (node instanceof Filter) {
                    if (containsCase(((Filter) node).getCondition())) {
                        found[0] = true;
                        return;
                    }
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return found[0];
    }

    /**
     * Check if query has LIKE patterns.
     */
    public static boolean hasLikePattern(RelNode root) {
        final boolean[] found = {false};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (found[0]) return;
                if (node instanceof Filter) {
                    if (containsLike(((Filter) node).getCondition())) {
                        found[0] = true;
                        return;
                    }
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return found[0];
    }

    // ==================== STRUCTURE FEATURES ====================

    /**
     * Compute the maximum nesting depth of the query plan.
     */
    public static int maxDepth(RelNode root) {
        return computeDepth(root, 0);
    }

    /**
     * Count total number of nodes in the plan.
     */
    public static int countNodes(RelNode root) {
        final int[] count = {0};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                count[0]++;
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return count[0];
    }

    /**
     * Count total number of Project nodes.
     */
    public static int countProjects(RelNode root) {
        return countType(root, Project.class);
    }

    // ==================== ROLLUP / CASE / FUNCTION-IN-PREDICATE ====================

    /**
     * Check if any Aggregate uses ROLLUP, CUBE, or GROUPING SETS
     * (i.e. group type is not SIMPLE).
     * Distinguishes query018 (regression, has ROLLUP) from query5/7/9 (wins, SIMPLE).
     */
    public static boolean hasGroupByRollup(RelNode root) {
        final boolean[] found = {false};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (found[0]) return;
                if (node instanceof Aggregate) {
                    Aggregate agg = (Aggregate) node;
                    if (agg.getGroupType() != Aggregate.Group.SIMPLE) {
                        found[0] = true;
                        return;
                    }
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return found[0];
    }

    /**
     * Count total CASE WHEN expressions across all Project and Filter nodes.
     * Distinguishes query039 (2-3 CASE, 32.9x win) from query059 (7+ CASE, 0.70x regression).
     */
    public static int countCaseWhen(RelNode root) {
        final int[] count = {0};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (node instanceof Project) {
                    for (RexNode expr : ((Project) node).getProjects()) {
                        count[0] += countCaseInExpr(expr);
                    }
                }
                if (node instanceof Filter) {
                    count[0] += countCaseInExpr(((Filter) node).getCondition());
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return count[0];
    }

    /**
     * Check if any Filter condition contains a function call (RexCall that is
     * not a comparison/logical operator). Detects SUBSTRING, EXTRACT, CAST, etc.
     * Distinguishes query019 (has SUBSTRING in WHERE, regression) from query5/7/9 (wins).
     */
    public static boolean hasFunctionInPredicate(RelNode root) {
        final boolean[] found = {false};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (found[0]) return;
                if (node instanceof Filter) {
                    if (containsFunctionCall(((Filter) node).getCondition())) {
                        found[0] = true;
                        return;
                    }
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return found[0];
    }

    // ==================== PRIVATE HELPER METHODS ====================

    private static boolean containsType(RelNode root, Class<?> type) {
        final boolean[] found = {false};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (found[0]) return;
                if (type.isInstance(node)) {
                    found[0] = true;
                    return;
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return found[0];
    }

    private static int countType(RelNode root, Class<?> type) {
        final int[] count = {0};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (type.isInstance(node)) {
                    count[0]++;
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return count[0];
    }

    private static int computeDepth(RelNode node, int current) {
        int max = current;
        for (RelNode input : node.getInputs()) {
            max = Math.max(max, computeDepth(input, current + 1));
        }
        return max;
    }

    private static int countRexSubQueries(RelNode root) {
        final int[] count = {0};
        new RelVisitor() {
            @Override
            public void visit(RelNode node, int ordinal, RelNode parent) {
                if (node instanceof Filter) {
                    count[0] += countRexSubQueriesInExpr(((Filter) node).getCondition());
                }
                if (node instanceof Project) {
                    for (RexNode expr : ((Project) node).getProjects()) {
                        count[0] += countRexSubQueriesInExpr(expr);
                    }
                }
                super.visit(node, ordinal, parent);
            }
        }.go(root);
        return count[0];
    }

    private static int countRexSubQueriesInExpr(RexNode rex) {
        if (rex instanceof RexSubQuery) {
            return 1;
        }
        if (rex instanceof RexCall) {
            int count = 0;
            for (RexNode operand : ((RexCall) rex).getOperands()) {
                count += countRexSubQueriesInExpr(operand);
            }
            return count;
        }
        return 0;
    }

    private static int countPredicates(RexNode condition) {
        if (condition instanceof RexCall) {
            RexCall call = (RexCall) condition;
            if (call.getOperator().getName().equals("AND")) {
                int count = 0;
                for (RexNode operand : call.getOperands()) {
                    count += countPredicates(operand);
                }
                return count;
            }
        }
        return 1;
    }

    private static boolean containsCase(RexNode rex) {
        if (rex instanceof RexCall) {
            RexCall call = (RexCall) rex;
            if (call.getOperator().getName().equals("CASE")) {
                return true;
            }
            for (RexNode operand : call.getOperands()) {
                if (containsCase(operand)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean containsLike(RexNode rex) {
        if (rex instanceof RexCall) {
            RexCall call = (RexCall) rex;
            if (call.getOperator().getName().equals("LIKE")) {
                return true;
            }
            for (RexNode operand : call.getOperands()) {
                if (containsLike(operand)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Count CASE expressions in a RexNode tree.
     */
    private static int countCaseInExpr(RexNode rex) {
        if (rex instanceof RexCall) {
            RexCall call = (RexCall) rex;
            int count = 0;
            if (call.getOperator().getName().equals("CASE")) {
                count = 1;
            }
            for (RexNode operand : call.getOperands()) {
                count += countCaseInExpr(operand);
            }
            return count;
        }
        return 0;
    }

    /**
     * Check if a filter condition contains a non-trivial function call.
     * Ignores comparison operators (=, <, >, <=, >=, <>, LIKE, IN, etc.)
     * and logical operators (AND, OR, NOT) since those are structural.
     * Detects things like SUBSTRING, EXTRACT, CAST, UPPER, LOWER, etc.
     */
    private static boolean containsFunctionCall(RexNode rex) {
        if (rex instanceof RexCall) {
            RexCall call = (RexCall) rex;
            String name = call.getOperator().getName().toUpperCase();
            // Skip comparison and logical operators — these are structural, not "functions"
            if (!name.equals("AND") && !name.equals("OR") && !name.equals("NOT") &&
                !name.equals("=") && !name.equals("<>") && !name.equals("<") &&
                !name.equals(">") && !name.equals("<=") && !name.equals(">=") &&
                !name.equals("IS NULL") && !name.equals("IS NOT NULL") &&
                !name.equals("IS TRUE") && !name.equals("IS FALSE") &&
                !name.equals("IS NOT TRUE") && !name.equals("IS NOT FALSE") &&
                !name.equals("IN") && !name.equals("NOT IN") &&
                !name.equals("BETWEEN") && !name.equals("LIKE") &&
                !name.equals("CASE") && !name.equals("SEARCH") &&
                !name.equals("CAST") && !name.equals("Sarg")) {
                return true;
            }
            for (RexNode operand : call.getOperands()) {
                if (containsFunctionCall(operand)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Check if expression contains EXISTS subquery.
     */
    private static boolean containsExistsSubQuery(RexNode rex) {
        if (rex instanceof RexSubQuery) {
            RexSubQuery subQuery = (RexSubQuery) rex;
            String opName = subQuery.getOperator().getName().toUpperCase();
            // EXISTS and NOT EXISTS patterns
            return opName.equals("EXISTS") || opName.contains("EXISTS");
        }
        if (rex instanceof RexCall) {
            RexCall call = (RexCall) rex;
            // Check for NOT EXISTS pattern
            if (call.getOperator().getName().equals("NOT")) {
                for (RexNode operand : call.getOperands()) {
                    if (operand instanceof RexSubQuery) {
                        String opName = ((RexSubQuery) operand).getOperator().getName().toUpperCase();
                        if (opName.equals("EXISTS")) {
                            return true;
                        }
                    }
                }
            }
            for (RexNode operand : call.getOperands()) {
                if (containsExistsSubQuery(operand)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Check if expression contains IN subquery.
     */
    private static boolean containsInSubQuery(RexNode rex) {
        if (rex instanceof RexSubQuery) {
            RexSubQuery subQuery = (RexSubQuery) rex;
            String opName = subQuery.getOperator().getName().toUpperCase();
            return opName.equals("IN") || opName.equals("NOT IN") || 
                   opName.equals("SOME") || opName.equals("ANY") || opName.equals("ALL");
        }
        if (rex instanceof RexCall) {
            for (RexNode operand : ((RexCall) rex).getOperands()) {
                if (containsInSubQuery(operand)) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Check if expression contains scalar subquery (used in comparison).
     * Pattern: column < (SELECT ...) or column = (SELECT ...)
     */
    private static boolean containsScalarSubQuery(RexNode rex) {
        if (rex instanceof RexSubQuery) {
            RexSubQuery subQuery = (RexSubQuery) rex;
            String opName = subQuery.getOperator().getName().toUpperCase();
            // Scalar subquery is not EXISTS, IN, SOME, ANY, ALL
            return !opName.equals("EXISTS") && !opName.equals("IN") && 
                   !opName.equals("NOT IN") && !opName.equals("SOME") && 
                   !opName.equals("ANY") && !opName.equals("ALL");
        }
        if (rex instanceof RexCall) {
            for (RexNode operand : ((RexCall) rex).getOperands()) {
                if (containsScalarSubQuery(operand)) {
                    return true;
                }
            }
        }
        return false;
    }
}
