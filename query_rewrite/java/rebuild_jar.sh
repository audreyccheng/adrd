#!/bin/bash
# Rebuild LearnedRewrite.jar with updated RuleSelector, QueryAnalyzer, and Rewriter

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

JAR_FILE="$SCRIPT_DIR/LearnedRewrite.jar"
TEMP_DIR="$SCRIPT_DIR/temp_compile"

# Source files (all in java/ directory)
RULESELECTOR_SRC="$SCRIPT_DIR/RuleSelector.java"
QUERYANALYZER_SRC="$SCRIPT_DIR/QueryAnalyzer.java"
REWRITER_SRC="$SCRIPT_DIR/Rewriter.java"

echo "=== Rebuilding LearnedRewrite.jar ==="
echo ""

# Check source files exist
for src in "$RULESELECTOR_SRC" "$QUERYANALYZER_SRC" "$REWRITER_SRC"; do
    if [ ! -f "$src" ]; then
        echo "ERROR: $(basename $src) not found at $src"
        exit 1
    fi
done

# Create temp compile directory
mkdir -p "$TEMP_DIR"

echo "1. Extracting class files from JAR for compilation..."
CLASSES_DIR="$TEMP_DIR/extracted_classes"
rm -rf "$CLASSES_DIR"
mkdir -p "$CLASSES_DIR"
cd "$CLASSES_DIR"
unzip -q -o "$JAR_FILE" '*.class' 2>/dev/null || true
cd "$TEMP_DIR"

# Create target package directory
HEP_DIR="$TEMP_DIR/org/apache/calcite/plan/hep"
REWRITER_DIR="$TEMP_DIR/rewriter"
mkdir -p "$HEP_DIR" "$REWRITER_DIR"

echo "2. Compiling RuleSelector.java and QueryAnalyzer.java..."
# Copy sources to package directory
cp "$RULESELECTOR_SRC" "$HEP_DIR/"
cp "$QUERYANALYZER_SRC" "$HEP_DIR/"

javac --release 17 -proc:none -cp "$CLASSES_DIR" -d "$TEMP_DIR" \
    "$HEP_DIR/QueryAnalyzer.java" "$HEP_DIR/RuleSelector.java"
echo "   Done"

echo "3. Compiling Rewriter.java..."
cp "$REWRITER_SRC" "$REWRITER_DIR/"
javac --release 17 -proc:none -cp "$CLASSES_DIR:$TEMP_DIR" -d "$TEMP_DIR" \
    "$REWRITER_DIR/Rewriter.java"
echo "   Done"

echo "4. Updating JAR file..."
cd "$TEMP_DIR"
jar uf "$JAR_FILE" org/apache/calcite/plan/hep/RuleSelector.class
for f in org/apache/calcite/plan/hep/QueryAnalyzer*.class; do
    [ -f "$f" ] && jar uf "$JAR_FILE" "$f"
done
for f in rewriter/Rewriter*.class; do
    [ -f "$f" ] && jar uf "$JAR_FILE" "$f"
done
echo "   Done"

echo ""
echo "5. Verifying..."
jar tf "$JAR_FILE" | grep -E "(RuleSelector|QueryAnalyzer|rewriter/Rewriter)" | head -10

# Cleanup
rm -rf "$CLASSES_DIR"

echo ""
echo "=== JAR rebuilt successfully ==="
echo "JAR: $JAR_FILE"
