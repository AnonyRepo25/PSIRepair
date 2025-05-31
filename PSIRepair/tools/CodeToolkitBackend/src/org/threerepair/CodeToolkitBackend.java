package org.threerepair;

import java.util.Set;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import java.util.stream.Collectors;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.charset.MalformedInputException;

import org.json.JSONObject;
import org.json.JSONArray;
import org.json.JSONTokener;
import spoon.Launcher;
import spoon.SpoonException;
import spoon.reflect.CtModel;
import spoon.reflect.code.CtComment;
import spoon.reflect.code.CtVariableAccess;
import spoon.reflect.code.CtAbstractInvocation;
import spoon.reflect.cu.SourcePosition;
import spoon.reflect.declaration.CtType;
import spoon.reflect.declaration.CtClass;
import spoon.reflect.declaration.CtElement;
import spoon.reflect.declaration.CtEnum;
import spoon.reflect.declaration.CtExecutable;
import spoon.reflect.declaration.CtInterface;
import spoon.reflect.declaration.CtMethod;
import spoon.reflect.declaration.CtConstructor;
import spoon.reflect.reference.CtFieldReference;
import spoon.reflect.reference.CtTypeReference;
import spoon.reflect.visitor.filter.TypeFilter;
import spoon.support.compiler.VirtualFile;

public class CodeToolkitBackend {
    public static String[] getPkgClsMdPathByPosition(String filePath, int startLine, int endLine) {
        return getPkgClsMdPathByPosition(filePath, startLine, endLine, false);
    }

    public static String[] getPkgClsMdPathByPosition(String filePath, int startLine, int endLine,
            boolean returnQualifiedName) {
        // I don't consider the column position, because I found that calculation
        // implemented by spoon is not accurate in some cases.

        var fPath = Paths.get(filePath).toAbsolutePath();

        if (filePath.isEmpty() || !Files.exists(fPath)) {
            throw new IllegalArgumentException(String.format("File %s does not exist", filePath));
        }

        Launcher launcher = new Launcher();
        launcher.addInputResource(filePath);
        launcher.getEnvironment().setAutoImports(true);
        launcher.getEnvironment().setCommentEnabled(true);
        launcher.buildModel();

        List<CtElement> elements = new ArrayList<>();
        elements.addAll(launcher.getModel().getElements(new TypeFilter<>(CtMethod.class)));
        elements.addAll(launcher.getModel().getElements(new TypeFilter<>(CtConstructor.class)));
        elements.addAll(launcher.getModel().getElements(new TypeFilter<>(CtType.class)));

        Optional<CtElement> minRangeElement = elements.stream().filter(e -> {
            // get the element contains the given range
            SourcePosition pos = e.getPosition();
            return (pos.getFile() != null && fPath.compareTo(Paths.get(pos.getFile().getAbsolutePath())) == 0) &&
                    pos.getLine() <= startLine && pos.getEndLine() >= endLine;
        }).min((e1, e2) -> {
            // get the element with the smallest range
            SourcePosition pos1 = e1.getPosition();
            SourcePosition pos2 = e2.getPosition();
            int diff1 = pos1.getSourceEnd() - pos1.getSourceStart();
            int diff2 = pos2.getSourceEnd() - pos2.getSourceStart();
            return diff1 - diff2;
        });

        if (!minRangeElement.isPresent()) {
            for (var type : launcher.getModel().getAllTypes()) {
                var typePos = type.getPosition();
                if (typePos.getFile() != null && fPath.compareTo(Paths.get(typePos.getFile().getAbsolutePath())) == 0) {
                    return new String[] { "package", type.getPackage().getQualifiedName() };
                }
            }
            throw new IllegalArgumentException(
                    String.format("The given range is not valid (%s:%d-%d))",
                            filePath, startLine, endLine));
        }

        CtElement element = minRangeElement.get();
        String pathType = null, path = null;
        if (element instanceof CtType) {
            if (element instanceof CtInterface) {
                pathType = "interface";
            } else if (element instanceof CtEnum) {
                pathType = "enum";
            } else {
                // maybe not class, but do not care
                pathType = "class";
            }
            path = ((CtType<?>) element).getQualifiedName();
        } else if (element instanceof CtMethod) {
            pathType = "method";
            var md = (CtMethod<?>) element;
            var declTy = md.getDeclaringType();
            assert declTy != null;
            if (returnQualifiedName) {
                path = String.format("%s.%s", declTy.getQualifiedName(), getSimpleSignatureOfMethodOrConstructor(md));
            } else {
                path = String.format("%s.%s", declTy.getQualifiedName(), md.getSimpleName());
            }
        } else if (element instanceof CtConstructor) {
            pathType = "method"; // do not differentiate constructor and method
            var md = (CtConstructor<?>) element;
            var declTy = md.getDeclaringType();
            assert declTy != null;
            // StringmdName=md.getType()!=null?md.getType().getSimpleName():md.getSimpleName();
            String mdName = md.getSimpleName(); // always "<init>"
            if (returnQualifiedName) {
                path = String.format("%s.%s", declTy.getQualifiedName(), getSimpleSignatureOfMethodOrConstructor(md));
            } else {
                path = String.format("%s.%s", declTy.getQualifiedName(), mdName);
            }
        }

        assert pathType != null && path != null;
        return new String[] { pathType, path };
    }

    public static String makeClassSkeleton(String classCode, boolean simplify) {
        return makeClassSkeleton(classCode, simplify, true, false);
    }

    public static String makeClassSkeleton(String classCode, boolean simplify, boolean cleanInternalComments,
            boolean removeMethods) {
        // Parse class code
        String wrappedClsCode = String.format("package do.not.care;\n\npublic class DoNotCare {\n\n%s\n\n}", classCode);
        Launcher launcher = new Launcher();
        launcher.getEnvironment().setNoClasspath(true);
        launcher.getEnvironment().setAutoImports(true);
        launcher.getEnvironment().setCommentEnabled(true);
        launcher.addInputResource(new VirtualFile(wrappedClsCode, "DoNotCare.java"));
        launcher.buildModel();
        CtType<?> class_ = launcher.getModel().getAllTypes().iterator().next().getNestedTypes().iterator().next();

        if (simplify) {
            String ty = (class_ instanceof CtInterface) ? "interface" : (class_ instanceof CtEnum) ? "enum" : "class";
            return String.format("%s %s { ... }", ty, class_.getSimpleName());
        }

        if (cleanInternalComments) {
            // Clean all comments except comment for the class
            var classComments = class_.getComments();
            for (var c : class_.getElements(new TypeFilter<>(CtComment.class))) {
                if (!classComments.contains(c)) {
                    c.delete();
                }
            }
        }

        String result = null;
        if (removeMethods) {
            // Remove all methods
            if (class_ instanceof CtClass<?> c) {
                for (var md : c.getConstructors()) {
                    md.delete();
                }
            }
            for (var md : class_.getMethods()) {
                md.delete();
            }
            // Remove all nested types
            for (var ty : class_.getNestedTypes()) {
                ty.delete();
            }
            result = class_.toString();
            /// Replace "}" with " ...\n}"
            int rightBraceIndex = result.lastIndexOf("}");
            assert rightBraceIndex != -1;
            result = result.substring(0, rightBraceIndex) + "\n\n    ...\n\n}"
                    + result.substring(rightBraceIndex + 1);
        } else {
            // Replace all method body with "// ..." comments
            if (class_ instanceof CtClass<?> c) {
                for (var md : c.getConstructors()) {
                    md.setBody(launcher.getFactory().createCodeSnippetStatement("/*ABCDEFGFEDCBA*/")); // it adds a addi
                                                                                                       // ';'
                }
            }
            for (var md : class_.getMethods()) {
                if (md.getBody() != null) { // not abstract method
                    md.setBody(launcher.getFactory().createCodeSnippetStatement("/*ABCDEFGFEDCBA*/")); // it adds a addi
                                                                                                       // ';'
                }
            }
            result = class_.toString().replaceAll("\\{\\s*/\\*ABCDEFGFEDCBA\\*/;\\s*\\}", "{ ... }");
        }

        return result;
    }

    public static String makeMethodSkeleton(String methodCode, boolean simplify) {
        CtElement md = parseMethodOrConstructor(methodCode);
        // Clean method comments
        for (var c : List.copyOf(md.getComments())) {
            md.removeComment(c);
        }
        if (!simplify) { // do not simplify, only clean comments
            return md.toString();
        }
        // Clean body
        if (md instanceof CtMethod method) {
            // it adds a addi ';'
            method.setBody(method.getFactory().Code().createCodeSnippetStatement("/*ABCDEFGFEDCBA*/"));
        } else if (md instanceof CtConstructor constructor) {
            // it adds a addi ';'
            constructor.setBody(constructor.getFactory().Code().createCodeSnippetStatement("/*ABCDEFGFEDCBA*/"));
        }
        return md.toString().replaceAll("\\{\\s*/\\*ABCDEFGFEDCBA\\*/;\\s*\\}", "{ ... }");
    }

    private static List<CtTypeReference<?>> getTypeAndComponentTypes(CtTypeReference<?> type) {
        List<CtTypeReference<?>> result = new ArrayList<>();
        result.add(type);
        if (type.getActualTypeArguments() != null) {
            for (var t : type.getActualTypeArguments()) {
                result.addAll(getTypeAndComponentTypes(t));
            }
        }
        return result;
    }

    private static String getSimpleSignatureOfMethodOrConstructor(CtElement mdOrCt) {
        if (mdOrCt instanceof CtMethod<?> md) {
            String parameters = String.join(",", md.getParameters().stream().map(e -> {
                return e.getType().getSimpleName();
            }).toArray(String[]::new));
            return String.format("%s(%s)", md.getSimpleName(), parameters);
        } else if (mdOrCt instanceof CtConstructor<?> md) {
            String parameters = String.join(",", md.getParameters().stream().map(e -> {
                return e.getType().getSimpleName();
            }).toArray(String[]::new));
            // StringmdName=md.getType()!=null?md.getType().getSimpleName():md.getSimpleName();
            String mdName = md.getSimpleName(); // always "<init>"
            return String.format("%s(%s)", mdName, parameters);
        }
        assert false;
        return null;
    }

    private static CtElement parseMethodOrConstructor(String code) {
        String classCode = String.format("class DummyClass {\n%s\n}", code);
        Launcher launcher = new Launcher();
        launcher.getEnvironment().setNoClasspath(true);
        launcher.getEnvironment().setAutoImports(true);
        launcher.getEnvironment().setCommentEnabled(true);
        launcher.addInputResource(new VirtualFile(classCode, "DoNotCare.java"));
        launcher.buildModel();
        CtClass<?> class_ = (CtClass<?>) launcher.getModel().getAllTypes().iterator().next();

        CtElement md = null;
        if (!class_.getMethods().isEmpty()) {
            assert class_.getMethods().size() == 1;
            md = class_.getMethods().iterator().next();
        } else if (!class_.getConstructors().isEmpty()) {
            assert class_.getConstructors().size() == 1;
            md = class_.getConstructors().iterator().next();
        } else {
            assert false;
        }
        return md;
    }

    public static CodeSeachEngine createCodeSearchEngine(String sourceRoots, String classRoots) {
        String[] srcRoots = sourceRoots.split(",");
        String[] clsRoots = classRoots.split(",");
        return createCodeSearchEngine(srcRoots, clsRoots, false);
    }

    public static CodeSeachEngine createCodeSearchEngine(String[] sourceRoots, String[] classRoots,
            boolean ignoreNestedTypes) {
        sourceRoots = Arrays.stream(sourceRoots).filter(p -> !p.isEmpty() && Files.exists(Paths.get(p)))
                .toArray(String[]::new);
        classRoots = Arrays.stream(classRoots).filter(p -> !p.isEmpty() && Files.exists(Paths.get(p)))
                .toArray(String[]::new);
        return new CodeSeachEngine(sourceRoots, classRoots, ignoreNestedTypes);
    }

    public static class ProjectGraph {
        public class Relationship {
            // Relationship between two nodes
            /// 1. Type-Type
            //// 1.1. inheritance
            //// 1.2. realization
            //// 1.3. association (including aggregation and composition)
            //// 1.4. dependency
            //// 1.5. contains (contains a nested type)
            // 2. Type-Method
            /// 2.1 contains
            // 3. Type-Fields
            /// 3.1 contains
            // 4. Method-Method
            /// 4.1. calls
            // 5. Method-Fields
            /// 5.1. references
            // 6. Fields-Fields
            /// ... No relationship between fields and fields

            static private final Set<String> availableTypes;
            static private final Map<String, String> typeToReadableTypeName;

            static {
                Set<String> _availableTypes = new HashSet<>();

                // 1. Type-Type
                _availableTypes.add("inheritance");
                _availableTypes.add("realization");
                _availableTypes.add("association");
                _availableTypes.add("dependency");
                _availableTypes.add("contains");
                // 2. Type-Method
                _availableTypes.add("contains");
                // 3. Type-Fields
                _availableTypes.add("contains");
                // 4. Method-Method
                _availableTypes.add("calls");
                // 5. Method-Fields
                _availableTypes.add("references");
                // Reversed Relationship Types
                _availableTypes.addAll(_availableTypes.stream().map(e -> String.format("r_%s", e)).toList());

                availableTypes = Collections.unmodifiableSet(_availableTypes);

                Map<String, String> _typeToReadableTypeName = new HashMap<>();
                _typeToReadableTypeName.put("inheritance", "extends");
                _typeToReadableTypeName.put("realization", "implements");
                _typeToReadableTypeName.put("association", "has association with");
                _typeToReadableTypeName.put("dependency", "depends on");
                _typeToReadableTypeName.put("contains", "contains");
                _typeToReadableTypeName.put("calls", "calls");
                _typeToReadableTypeName.put("references", "references");
                _typeToReadableTypeName.put("r_inheritance", "extended by");
                _typeToReadableTypeName.put("r_realization", "implemented by");
                _typeToReadableTypeName.put("r_association", "associated in");
                _typeToReadableTypeName.put("r_dependency", "depended on by");
                _typeToReadableTypeName.put("r_contains", "contained by");
                _typeToReadableTypeName.put("r_calls", "called by");
                _typeToReadableTypeName.put("r_references", "referenced by");
                typeToReadableTypeName = Collections.unmodifiableMap(_typeToReadableTypeName);
            }

            // type
            public String type;

            // from, to
            public Node from;
            public Node to;

            public Relationship(String type, Node from, Node to) {
                if (!availableTypes.contains(type)) {
                    throw new IllegalArgumentException(String.format("unsupported relationship: %s", type));
                }

                this.type = type;
                this.from = from;
                this.to = to;
            }

            @Override
            public int hashCode() {
                /* type, from, and to */

                final int prime = 31;
                int result = 1;
                result = prime * result + getEnclosingInstance().hashCode();
                result = prime * result + ((type == null) ? 0 : type.hashCode());
                result = prime * result + ((from == null) ? 0 : from.hashCode());
                result = prime * result + ((to == null) ? 0 : to.hashCode());
                return result;
            }

            @Override
            public boolean equals(Object obj) {
                /* type, from, and to */

                if (this == obj)
                    return true;
                if (obj == null)
                    return false;
                if (getClass() != obj.getClass())
                    return false;
                Relationship other = (Relationship) obj;
                if (!getEnclosingInstance().equals(other.getEnclosingInstance()))
                    return false;
                if (type == null) {
                    if (other.type != null)
                        return false;
                } else if (!type.equals(other.type))
                    return false;
                if (from == null) {
                    if (other.from != null)
                        return false;
                } else if (!from.equals(other.from))
                    return false;
                if (to == null) {
                    if (other.to != null)
                        return false;
                } else if (!to.equals(other.to))
                    return false;
                return true;
            }

            private ProjectGraph getEnclosingInstance() {
                return ProjectGraph.this;
            }

            public String getReadableTypeName() {
                return typeToReadableTypeName.get(type);
            }
        }

        public class Node {
            /* Unified Representation of Type, Method, and Fields Node */

            // Meta information
            public String type; // type, method, or fields
            public String detailedType; // class, interface, enum, method, or fields
            public String path; // path for type or method, class path for fields
            public String qualifiedName; // path for type, qualified name for method, class path for fields
            public String filePath; // for type, method, and fields
            public String code; // for type, method, and fields

            // Relationships
            public Set<Relationship> relationships = new HashSet<Relationship>();

            // Extended information only for serialization
            public Integer indexInNodePool = null;

            public void addRelationship(String type, Node to) {
                if (to == null) {
                    throw new IllegalArgumentException("Node to cannot be null");
                }

                // Set<String> rels = new HashSet<>();
                // rels.add("association");
                // rels.add("calls");
                // rels.add("contains");
                // rels.add("dependency");
                // rels.add("inheritance");
                // rels.add("realization");
                // rels.add("references");
                // if (!rels.contains(type)) return;

                relationships.add(new Relationship(type, this, to)); // this -> to
                to.relationships.add(new Relationship("r_" + type, to, this)); // to -> this
            }

            @Override
            public int hashCode() {
                /* type and qualifiedName */

                final int prime = 31;
                int result = 1;
                result = prime * result + getEnclosingInstance().hashCode();
                result = prime * result + ((type == null) ? 0 : type.hashCode());
                result = prime * result + ((qualifiedName == null) ? 0 : qualifiedName.hashCode());
                return result;
            }

            @Override
            public boolean equals(Object obj) {
                /* type and qualifiedName */

                if (this == obj)
                    return true;
                if (obj == null)
                    return false;
                if (getClass() != obj.getClass())
                    return false;
                Node other = (Node) obj;
                if (!getEnclosingInstance().equals(other.getEnclosingInstance()))
                    return false;
                if (type == null) {
                    if (other.type != null)
                        return false;
                } else if (!type.equals(other.type))
                    return false;
                if (qualifiedName == null) {
                    if (other.qualifiedName != null)
                        return false;
                } else if (!qualifiedName.equals(other.qualifiedName))
                    return false;
                return true;
            }

            private ProjectGraph getEnclosingInstance() {
                return ProjectGraph.this;
            }

            public String getReadableName() {
                if (type.equals("type")) {
                    // class/interface/enum path.to...
                    return String.format("%s %s", detailedType, path);
                } else if (type.equals("method")) {
                    // path.to...type.methodName(...)
                    return String.format("%s", qualifiedName);
                } else if (type.equals("fields")) {
                    // fields of class path.to...
                    return String.format("fields of %s %s", detailedType, path);
                } else {
                    throw new IllegalArgumentException(String.format("unsupported type: %s", type));
                }
            }
        }

        // Meta information
        private String[] sourceRoots;
        private String[] classRoots;

        // Nodes, key: Node.qualifiedName
        private Map<String, Node> typeNodes;
        private Map<String, Node> fieldsNodes;
        private Map<String, Node> methodNodes;

        private static void debugPrintln(String format, Object... obj) {
            if (isDebugMode()) {
                StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
                if (stackTrace.length >= 3) {
                    StackTraceElement caller = stackTrace[2];
                    String callerInfo = String.format("%s:%d@%s",
                            caller.getFileName(),
                            caller.getLineNumber(),
                            caller.getMethodName());
                    String string = String.format(format, obj);
                    for (var line : string.split("\n")) {
                        System.out.printf("[ProjectGraph:debug:%s] %s \n", callerInfo, line.stripTrailing());
                    }
                } else {
                    System.out.println("[ProjectGraph:debug] Unable to determine caller information.");
                }
            }
        }

        private static boolean isDebugMode() {
            return "1".equals(System.getenv("DEBUG"));
        }

        public static ProjectGraph load(String filePath) throws IOException {

            class Helper {
                private static void populateNodeMap(Map<String, Node> map, JSONArray indexes, List<Node> nodePool) {
                    for (int i = 0; i < indexes.length(); i++) {
                        Node node = nodePool.get(indexes.getInt(i));
                        map.put(node.qualifiedName, node); // qualifiedName as the key
                    }
                }

                private static List<String> jsonArrayToList(JSONArray jsonArray) {
                    List<String> list = new ArrayList<>(jsonArray.length());
                    for (int i = 0; i < jsonArray.length(); i++) {
                        list.add(jsonArray.getString(i));
                    }
                    return list;
                }
            }

            ProjectGraph graph = new ProjectGraph();

            JSONObject root;
            try (FileReader reader = new FileReader(filePath)) {
                root = new JSONObject(new JSONTokener(reader));
            }

            // Parse sourceRoots and classRoots
            List<String> sourceRoots = Helper.jsonArrayToList(root.getJSONArray("sourceRoots"));
            List<String> classRoots = Helper.jsonArrayToList(root.getJSONArray("classRoots"));

            // Build node pool
            JSONArray nodesArray = root.getJSONArray("nodes");
            List<Node> nodePool = new ArrayList<>(nodesArray.length());
            for (int i = 0; i < nodesArray.length(); i++) {
                JSONObject nodeJson = nodesArray.getJSONObject(i);
                Node node = graph.new Node();
                node.type = nodeJson.getString("type");
                node.detailedType = nodeJson.getString("detailedType");
                node.path = nodeJson.getString("path");
                node.qualifiedName = nodeJson.getString("qualifiedName");
                node.filePath = nodeJson.optString("filePath", null);
                node.code = nodeJson.getString("code");
                node.indexInNodePool = nodeJson.getInt("indexInNodePool");
                node.relationships = new HashSet<>();
                nodePool.add(node);
            }

            // Build relationships
            for (int i = 0; i < nodesArray.length(); i++) {
                JSONObject nodeJson = nodesArray.getJSONObject(i);
                Node currentNode = nodePool.get(i);
                JSONArray relationships = nodeJson.getJSONArray("relationships");

                for (int j = 0; j < relationships.length(); j++) {
                    JSONObject relJson = relationships.getJSONObject(j);
                    var type = relJson.getString("type");
                    var from = nodePool.get(relJson.getInt("from"));
                    var to = nodePool.get(relJson.getInt("to"));
                    Relationship rel = graph.new Relationship(type, from, to);
                    currentNode.relationships.add(rel);
                }
            }

            // Initialize the meta information
            graph.sourceRoots = sourceRoots.toArray(new String[0]);
            graph.classRoots = classRoots.toArray(new String[0]);

            // Populate the node maps
            Map<String, Node> typeNodes = new HashMap<>();
            Map<String, Node> fieldsNodes = new HashMap<>();
            Map<String, Node> methodNodes = new HashMap<>();
            Helper.populateNodeMap(typeNodes, root.getJSONArray("typeNodeIndexes"), nodePool);
            Helper.populateNodeMap(fieldsNodes, root.getJSONArray("fieldsNodeIndexes"), nodePool);
            Helper.populateNodeMap(methodNodes, root.getJSONArray("methodNodeIndexes"), nodePool);
            graph.typeNodes = Collections.unmodifiableMap(typeNodes);
            graph.fieldsNodes = Collections.unmodifiableMap(fieldsNodes);
            graph.methodNodes = Collections.unmodifiableMap(methodNodes);

            return graph;
        }

        public static ProjectGraph build(String[] sourceRoots, String[] classRoots) {
            try {
                return _build(sourceRoots, classRoots);
            } catch (Exception ex) {
                ex.printStackTrace();
                throw ex;
            }
        }

        public static ProjectGraph _build(String[] sourceRoots, String[] classRoots) {
            debugPrintln("Building Project Graph ...");
            debugPrintln(">>>> Added source roots: %s", Arrays.toString(sourceRoots));
            debugPrintln(">>>> Added class roots: %s", Arrays.toString(classRoots));
            ProjectGraph graph = new ProjectGraph();
            Map<String, Node> allTypeNodes = new HashMap<String, Node>();
            Map<String, Node> allFieldsNodes = new HashMap<String, Node>();
            Map<String, Node> allMethodNodes = new HashMap<String, Node>();

            debugPrintln(">>>> Indexing Project ...");
            CodeSeachEngine index = createCodeSearchEngine(sourceRoots, classRoots, false);

            // Collect nodes & Connect Type-Fields and Type-Method
            debugPrintln(">>>> Collecting nodes & Connecting Type-Fields and Type-Method ...");
            for (var cls : index.getClasses()) {
                // type node
                Node node = graph.new Node();
                node.type = "type";
                node.detailedType = cls.type;
                node.path = cls.path;
                node.qualifiedName = cls.path;
                node.filePath = cls.filePath;
                node.code = cls.code;
                allTypeNodes.put(node.qualifiedName, node);
                // fields node
                Node fieldsNode = graph.new Node();
                fieldsNode.type = "fields";
                fieldsNode.detailedType = "fields";
                fieldsNode.path = cls.path;
                fieldsNode.qualifiedName = cls.path;
                fieldsNode.filePath = cls.filePath;
                fieldsNode.code = makeClassSkeleton(cls.code, false, false, true);
                allFieldsNodes.put(fieldsNode.qualifiedName, fieldsNode);
                // method nodes
                Map<String, Node> methodNodes = index.getMethods().stream().filter(m -> m.classPath.equals(cls.path))
                        .map(m -> {
                            Node methodNode = graph.new Node();
                            methodNode.type = "method";
                            methodNode.detailedType = "method";
                            methodNode.path = m.path;
                            methodNode.qualifiedName = m.qualifiedName;
                            methodNode.filePath = m.filePath;
                            methodNode.code = m.code;
                            return methodNode;
                        }).collect(Collectors.toMap(m -> m.qualifiedName, m -> m, (a, b) -> a));
                allMethodNodes.putAll(methodNodes);
                // relationship between type and fields
                node.addRelationship("contains", fieldsNode);
                // relationships between type and methods
                for (var methodNode : methodNodes.values()) {
                    node.addRelationship("contains", methodNode);
                }
            }
            debugPrintln(">>>>>> Collected %d type nodes", allTypeNodes.size());
            debugPrintln(">>>>>> Collected %d fields nodes", allFieldsNodes.size());
            debugPrintln(">>>>>> Collected %d method nodes", allMethodNodes.size());

            // Connect Type-Type
            debugPrintln(">>>> Connecting Type-Type ...");
            for (var cls : index.getClasses()) {
                CtType<?> ctType = cls.ctType;
                Node node = allTypeNodes.get(cls.path);
                assert node != null;

                // inheritance
                CtTypeReference<?> superClassRef = ctType.getSuperclass();
                if (superClassRef != null) {
                    String superClassPath = superClassRef.getQualifiedName();
                    Node superClassNode = allTypeNodes.get(superClassPath);
                    if (superClassNode != null) {
                        node.addRelationship("inheritance", superClassNode);
                    }
                }

                // realization
                for (CtTypeReference<?> interfaceRef : ctType.getSuperInterfaces()) {
                    String interfacePath = interfaceRef.getQualifiedName();
                    Node interfaceNode = allTypeNodes.get(interfacePath);
                    if (interfaceNode != null) {
                        node.addRelationship("realization", interfaceNode);
                    }
                }

                // association (including aggregation and composition)
                Set<String> associatedTypes = new HashSet<String>();
                for (var field : ctType.getFields()) {
                    for (CtTypeReference<?> typeRef : getTypeAndComponentTypes(field.getType())) {
                        String typePath = typeRef.getQualifiedName();
                        if (!associatedTypes.contains(typePath)) { // not added
                            associatedTypes.add(typePath);
                            Node typeNode = allTypeNodes.get(typePath);
                            if (typeNode != null) {
                                node.addRelationship("association", typeNode);
                            }
                        }
                    }
                }

                // dependency
                Set<String> dependencyTypes = new HashSet<String>();
                for (var method : ctType.getMethods()) {
                    // Collect dependency types from method
                    List<CtTypeReference<?>> typeRefs = new ArrayList<>();
                    /// Parameters
                    typeRefs.addAll(
                            method.getParameters().stream().flatMap(p -> getTypeAndComponentTypes(p.getType()).stream())
                                    .collect(Collectors.toList()));
                    /// Return type
                    typeRefs.addAll(getTypeAndComponentTypes(method.getType()));
                    /// Throws
                    typeRefs.addAll(method.getThrownTypes());
                    /// Referenced in body
                    if (method.getBody() != null) {
                        typeRefs.addAll(method.getBody().getReferencedTypes());
                    }

                    // Add dependency types
                    for (var typeRef : typeRefs) {
                        String typePath = typeRef.getQualifiedName();
                        // - not added ----------------------- and not a type in associatedTypes
                        if (!dependencyTypes.contains(typePath) && !associatedTypes.contains(typePath)) {
                            dependencyTypes.add(typePath);
                            Node typeNode = allTypeNodes.get(typePath);
                            if (typeNode != null) {
                                node.addRelationship("dependency", typeNode);
                            }
                        }
                    }
                }

                // contains (contains a nested type)
                for (CtType<?> nestedType : ctType.getNestedTypes()) {
                    String nestedTypePath = nestedType.getQualifiedName();
                    Node nestedTypeNode = allTypeNodes.get(nestedTypePath);
                    if (nestedTypeNode != null) {
                        node.addRelationship("contains", nestedTypeNode);
                    }
                }
            }

            // Connect Method-Method
            debugPrintln(">>>> Connecting Method-Method ...");
            for (var md : index.getMethods()) {
                var ctMethod = md.ctMethod;
                Node node = allMethodNodes.get(md.qualifiedName);

                if (node == null) {
                    continue;
                }

                // calls
                List<Node> calledMethodNodes = ctMethod.getElements(new TypeFilter<>(CtAbstractInvocation.class))
                        .stream().filter(e -> {
                            return e.getExecutable().getDeclaration() != null
                                    && e.getExecutable().getDeclaringType() != null;
                        }).map(e -> {
                            var execRef = e.getExecutable();
                            var exec = execRef.getDeclaration();
                            var declaringType = execRef.getDeclaringType();
                            assert exec != null && declaringType != null;
                            String declClassPath = declaringType.getQualifiedName();
                            String simpleSignature = getSimpleSignatureOfMethodOrConstructor(exec);
                            String qualifiedName = String.format("%s.%s", declClassPath, simpleSignature);
                            return allMethodNodes.get(qualifiedName);
                        }).filter(e -> e != null).toList();
                for (var calledMethodNode : calledMethodNodes) {
                    node.addRelationship("calls", calledMethodNode);
                }
            }

            // Connect Method-Fields
            debugPrintln(">>>> Connecting Method-Fields ...");
            for (var md : index.getMethods()) {
                var ctMethod = md.ctMethod;
                Node node = allMethodNodes.get(md.qualifiedName);

                if (node == null) {
                    continue;
                }

                // references
                if (ctMethod.getBody() != null) {
                    List<Node> referencedFieldNodes = ctMethod.getBody()
                            .getElements(new TypeFilter<>(CtVariableAccess.class)).stream().map(e -> {
                                if (e.getVariable() instanceof CtFieldReference fieldRef) {
                                    var definedIn = fieldRef.getDeclaringType();
                                    if (definedIn != null) {
                                        String definedInPath = definedIn.getQualifiedName();
                                        return allFieldsNodes.get(definedInPath);
                                    }
                                }
                                return null;
                            }).filter(e -> e != null).toList();
                    for (var referencedFieldNode : referencedFieldNodes) {
                        node.addRelationship("references", referencedFieldNode);
                    }
                }
            }

            // Log relationship stat
            if (isDebugMode()) {
                debugPrintln(">>>> Stat Relationships ...");
                final Map<String, Integer> cntOfRel = new TreeMap<String, Integer>();
                for (var relType : Relationship.availableTypes) {
                    cntOfRel.put(relType, 0);
                }
                allTypeNodes.values().stream()
                        .flatMap(e -> e.relationships.stream())
                        .forEach(e -> cntOfRel.replace(e.type, cntOfRel.get(e.type) + 1));
                allMethodNodes.values().stream()
                        .flatMap(e -> e.relationships.stream())
                        .forEach(e -> cntOfRel.replace(e.type, cntOfRel.get(e.type) + 1));
                allFieldsNodes.values().stream()
                        .flatMap(e -> e.relationships.stream())
                        .forEach(e -> cntOfRel.replace(e.type, cntOfRel.get(e.type) + 1));
                for (var kv : cntOfRel.entrySet()) {
                    if (!kv.getKey().startsWith("r_")) {
                        var rRel = String.format("r_%s", kv.getKey());
                        debugPrintln(">>>>>> Num of rel '%s': %d", kv.getKey(), kv.getValue().intValue());
                        debugPrintln(">>>>>> Num of rel '%s': %d", rRel, cntOfRel.get(rRel).intValue());
                    }
                }
            }

            // Set meta information
            graph.sourceRoots = sourceRoots;
            graph.classRoots = classRoots;
            // Set populated nodes
            graph.typeNodes = Collections.unmodifiableMap(allTypeNodes);
            graph.fieldsNodes = Collections.unmodifiableMap(allFieldsNodes);
            graph.methodNodes = Collections.unmodifiableMap(allMethodNodes);
            return graph;
        }

        private ProjectGraph() {
        }

        public void save(String path) throws IOException {
            JSONObject root = new JSONObject();
            // Meta information
            root.put("sourceRoots", Arrays.asList(sourceRoots));
            root.put("classRoots", Arrays.asList(classRoots));
            // Kinds of nodes
            JSONArray nodes = new JSONArray();
            List<Node> nodePool = new ArrayList<Node>();
            nodePool.addAll(this.typeNodes.values());
            nodePool.addAll(this.fieldsNodes.values());
            nodePool.addAll(this.methodNodes.values());
            List<Integer> typeNodeIndexes = null;
            List<Integer> fieldsNodeIndexes = null;
            List<Integer> methodNodeIndexes = null;
            for (int i = 0; i < nodePool.size(); i++) {
                Node node = nodePool.get(i);
                node.indexInNodePool = i;
            }
            typeNodeIndexes = this.typeNodes.values().stream().map(e -> e.indexInNodePool).toList();
            fieldsNodeIndexes = this.fieldsNodes.values().stream().map(e -> e.indexInNodePool).toList();
            methodNodeIndexes = this.methodNodes.values().stream().map(e -> e.indexInNodePool).toList();
            root.put("typeNodeIndexes", typeNodeIndexes);
            for (var node : nodePool) {
                // if (!(node.type != null && node.detailedType != null && node.path != null
                // && node.qualifiedName != null && node.filePath != null && node.code != null
                // && node.relationships != null && node.indexInNodePool != null)) {
                // System.out.println(node.type);
                // System.out.println(node.detailedType);
                // System.out.println(node.path);
                // System.out.println(node.qualifiedName);
                // System.out.println(node.filePath);
                // System.out.println(node.code);
                // System.out.println(node.relationships != null ? node.relationships.size() :
                // "null");
                // System.out.println(node.indexInNodePool);
                // }

                assert node.type != null && node.detailedType != null && node.path != null
                        && node.qualifiedName != null && node.code != null
                        && node.relationships != null && node.indexInNodePool != null;

                JSONObject nodeJson = new JSONObject();

                // Meta information
                nodeJson.put("type", node.type);
                nodeJson.put("detailedType", node.detailedType);
                nodeJson.put("path", node.path);
                nodeJson.put("qualifiedName", node.qualifiedName);
                nodeJson.put("filePath", node.filePath != null ? node.filePath : JSONObject.NULL);
                nodeJson.put("code", node.code);
                // Relationships
                JSONArray relationships = new JSONArray();
                for (var rel : node.relationships) {
                    JSONObject relJson = new JSONObject();
                    relJson.put("type", rel.type);
                    relJson.put("from", rel.from.indexInNodePool);
                    relJson.put("to", rel.to.indexInNodePool);
                    relationships.put(relJson);
                }
                nodeJson.put("relationships", relationships);
                // Extended information
                nodeJson.put("indexInNodePool", node.indexInNodePool);

                nodes.put(nodeJson);
            }
            root.put("nodes", nodes);
            root.put("typeNodeIndexes", typeNodeIndexes);
            root.put("fieldsNodeIndexes", fieldsNodeIndexes);
            root.put("methodNodeIndexes", methodNodeIndexes);
            try (FileWriter file = new FileWriter(path)) {
                file.write(root.toString());
                file.flush();
            }
        }

        public String[] getSourceRoots() {
            return sourceRoots;
        }

        public String[] getClassRoots() {
            return classRoots;
        }

        public Map<String, Node> getTypeNodes() {
            return typeNodes;
        }

        public Map<String, Node> getFieldsNodes() {
            return fieldsNodes;
        }

        public Map<String, Node> getMethodNodes() {
            return methodNodes;
        }

        public Node getTypeNode(String qualifiedName) {
            // class/interface/enum path
            return typeNodes.get(qualifiedName);
        }

        public Node getFieldsNode(String qualifiedName) {
            // class path
            return fieldsNodes.get(qualifiedName);
        }

        public Node getMethodNode(String qualifiedName) {
            // class path + method name + parameter type names
            return methodNodes.get(qualifiedName);
        }

        public int getMaxBFSDepth(Node node) {
            if (node == null) {
                throw new IllegalArgumentException("Around node cannot be null.");
            }

            Set<Node> visited = new HashSet<>();
            Queue<Node> queue = new LinkedList<>();

            queue.add(node);
            visited.add(node);

            int currentDepth = 0;

            while (!queue.isEmpty()) {
                int levelSize = queue.size();
                for (int i = 0; i < levelSize; i++) {
                    Node currentNode = queue.poll();
                    for (Relationship rel : currentNode.relationships) {
                        Node neighbor = rel.to;
                        if (!visited.contains(neighbor)) {
                            queue.add(neighbor);
                            visited.add(neighbor);
                        }
                    }
                }
                currentDepth++;
            }

            return currentDepth - 1;
        }

        public class ReturnedNodeOfGetNodesFromEgoGraph {
            public Node node;
            public int depth;
            public List<Relationship> forwardRelationships; // from ego to this
            public String forwardRelationshipsString; // from ego to this
        }

        public List<ReturnedNodeOfGetNodesFromEgoGraph> getNodesFromEgoGraph(Node ego, int numFieldsNodes,
                int numMethodNodes) {
            if (ego == null) {
                throw new IllegalArgumentException("Around node cannot be null.");
            }
            if (numFieldsNodes < 0 || numMethodNodes < 0) {
                throw new IllegalArgumentException("Number of fields and method nodes must be non-negative.");
            }

            class TraversalNode {
                public Node node;
                public int depth;
                public Relationship backwardRelationship; // from backwardNode to this (not strict "backward"
                public TraversalNode backwardNode;

                public TraversalNode(
                        Node node,
                        int depth,
                        Relationship backwardRelationship,
                        TraversalNode backwardNode) {
                    this.node = node;
                    this.depth = depth;
                    this.backwardRelationship = backwardRelationship;
                    this.backwardNode = backwardNode;
                }

                @Override
                public int hashCode() {
                    return node.hashCode();
                }

                @Override
                public boolean equals(Object obj) {
                    return obj instanceof TraversalNode other && other.node.equals(node);
                }
            }

            Set<Node> visited = new HashSet<>();
            Queue<TraversalNode> queue = new LinkedList<>();
            List<TraversalNode> collectedNodes = new ArrayList<>();
            int countOfFieldsNodes = 0, countOfMethodNodes = 0;

            queue.add(new TraversalNode(ego, 0, null, null));
            visited.add(ego);

            while ((countOfFieldsNodes < numFieldsNodes ||
                    countOfMethodNodes < numMethodNodes) &&
                    !queue.isEmpty()) {
                int levelSize = queue.size();
                int currentNumOfFieldsNodes = countOfFieldsNodes; // current number of fields nodes before this level
                int currentNumOfMethodNodes = countOfMethodNodes; // current number of method nodes before this level

                for (int i = 0; i < levelSize; i++) {
                    TraversalNode wNode = queue.poll();
                    Node node = wNode.node;
                    if (node != ego && // skip the ego node
                            ((currentNumOfFieldsNodes < numFieldsNodes && node.type.equals("fields")) ||
                                    (currentNumOfMethodNodes < numMethodNodes && node.type.equals("method")))) {
                        assert visited.contains(node);
                        collectedNodes.add(wNode);
                        countOfFieldsNodes += node.type.equals("fields") ? 1 : 0;
                        countOfMethodNodes += node.type.equals("method") ? 1 : 0;
                    }
                    for (Relationship rel : node.relationships) {
                        Node neighbor = rel.to;
                        if (!visited.contains(neighbor)) {
                            queue.add(new TraversalNode(neighbor, wNode.depth + 1, rel, wNode));
                            visited.add(neighbor);
                        }
                    }
                }
            }

            assert queue.isEmpty()
                    || collectedNodes.stream().filter(e -> e.node.type.equals("fields")).count() >= numFieldsNodes;
            assert queue.isEmpty()
                    || collectedNodes.stream().filter(e -> e.node.type.equals("method")).count() >= numMethodNodes;

            return collectedNodes.stream().map(e -> {
                var returnedNode = new ReturnedNodeOfGetNodesFromEgoGraph();
                returnedNode.node = e.node;
                returnedNode.depth = e.depth;
                returnedNode.forwardRelationships = new LinkedList<>();
                var wNode = e;
                while (wNode.backwardNode != null) {
                    returnedNode.forwardRelationships.add(0, wNode.backwardRelationship);
                    wNode = wNode.backwardNode;
                }
                StringBuilder forwardRelationshipsStringSb = new StringBuilder();
                if (returnedNode.forwardRelationships.size() > 0) {
                    forwardRelationshipsStringSb
                            .append(returnedNode.forwardRelationships.get(0).from.getReadableName());
                }
                for (var rel : returnedNode.forwardRelationships) {
                    forwardRelationshipsStringSb.append(" --(");
                    forwardRelationshipsStringSb.append(rel.getReadableTypeName());
                    forwardRelationshipsStringSb.append(")--> ");
                    forwardRelationshipsStringSb.append(rel.to.getReadableName());
                }
                returnedNode.forwardRelationshipsString = forwardRelationshipsStringSb.toString();
                assert returnedNode.depth > 0; // skipped the ego node
                assert returnedNode.forwardRelationships.size() == returnedNode.depth;
                assert e.depth > 0 && returnedNode.forwardRelationships.get(0).from == ego;
                assert e.depth > 0 && returnedNode.forwardRelationships.get(e.depth - 1).to == returnedNode.node;
                return returnedNode;
            }).sorted(
                    (a, b) -> {
                        int depthDiff = a.depth - b.depth;
                        if (depthDiff != 0) {
                            return depthDiff;
                        }
                        return a.node.qualifiedName.compareTo(b.node.qualifiedName);
                    }).toList();
        }

        public List<Node> getSubgraphAround(Node around, int depth, boolean extendTypeNodes) {
            if (around == null) {
                throw new IllegalArgumentException("Around node cannot be null.");
            }
            if (depth < 0) {
                throw new IllegalArgumentException("Depth must be non-negative.");
            }

            Set<Node> visited = new HashSet<>();
            Queue<Node> queue = new LinkedList<>();

            queue.add(around);
            visited.add(around);

            int currentDepth = 0;

            while (currentDepth < depth && !queue.isEmpty()) {
                int levelSize = queue.size();
                for (int i = 0; i < levelSize; i++) {
                    Node node = queue.poll();
                    for (Relationship rel : node.relationships) {
                        Node neighbor = rel.to;
                        if (!visited.contains(neighbor)) {
                            queue.add(neighbor);
                            visited.add(neighbor);
                        }
                    }
                }
                currentDepth++;
            }

            if (extendTypeNodes) {
                Set<Node> extended = visited.stream()
                        // type nodes
                        .filter(node -> node.type.equals("type"))
                        .flatMap(node -> node.relationships.stream())
                        // contains (fields) and (method)s
                        .filter(rel -> rel.type.equals("contains")
                                && ("fields".equals(rel.to.type) || "method".equals(rel.to.type)))
                        .map(rel -> rel.to)
                        // exclude visited
                        .filter(toNode -> !visited.contains(toNode))
                        .collect(Collectors.toSet());

                visited.addAll(extended);
            }

            return new ArrayList<>(visited);
        }
    }

    public static class CodeSeachEngine {
        public class Class {
            public String type; // class, interface, or enum
            public String name;
            public String path; // full path
            public String code;
            public String filePath;
            private CtType<?> ctType;
        }

        public class Method {
            public String name;
            public String path; // full path
            public String classPath; // full path of class the method belongs to
            public String qualifiedName; // full path + parameter types
            public String code;
            public String filePath;
            public int startLine;
            public int startColumn;
            public int endLine;
            public int endColumn;
            private CtExecutable<?> ctMethod; // CtMethod or CtConstructor
        }

        public class CodeSnippet {
            public String path; // full path of class or method the snippet belongs to
            public String pathType; // class or method
            public String code;

            public String filePath;
            public List<Integer> lineNumbers = new ArrayList<Integer>();
            public List<String> lines = new ArrayList<String>();
        }

        private final String[] sourceRoots;
        private final String[] classRoots;
        private List<Class> classes;
        private List<Method> methods;

        public CodeSeachEngine(String[] sourceRoots, String[] classRoots, boolean ignoreNestedTypes) {
            this.sourceRoots = sourceRoots;
            this.classRoots = classRoots;
            this.buildIndex(ignoreNestedTypes);
        }

        private static void debugPrintln(String format, Object... obj) {
            if ("1".equals(System.getenv("DEBUG"))) {
                StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
                if (stackTrace.length >= 3) {
                    StackTraceElement caller = stackTrace[2];
                    String callerInfo = String.format("%s:%d@%s",
                            caller.getFileName(),
                            caller.getLineNumber(),
                            caller.getMethodName());
                    String string = String.format(format, obj);
                    for (var line : string.split("\n")) {
                        System.out.printf("[CodeSeachEngine:debug:%s] %s \n", callerInfo, line.stripTrailing());
                    }
                } else {
                    System.out.println("[CodeSeachEngine:debug] Unable to determine caller information.");
                }
            }
        }

        private static String getSourcCodeOfElement(CtElement e) {
            try {
                return e.getOriginalSourceFragment().getSourceCode();
            } catch (RuntimeException ex) {
                if (!(ex instanceof SpoonException) && !(ex.getCause() instanceof MalformedInputException)) {
                    System.out.println("\033[31mWARNING: Unexpected exception when getting source code\033[0m");
                    ex.printStackTrace();
                }
                return e.toString();
            }
        };

        private static Stream<CtType<?>> getAllTopLevelAndNestedTypes(CtModel model) {
            Stream<CtType<?>> types = model.getAllTypes().stream();
            Stream<CtType<?>> nestedTypes = model.getAllTypes().stream().flatMap(t -> t.getNestedTypes().stream());
            return Stream.concat(types, nestedTypes);
        }

        private void buildIndex(boolean ignoreNestedTypes) {
            debugPrintln("Building index...");
            debugPrintln(">>>> Adding source roots: %s", Arrays.toString(sourceRoots));
            debugPrintln(">>>> Adding class roots: %s", Arrays.toString(classRoots));

            Launcher launcher = new Launcher();
            for (String root : sourceRoots) {
                launcher.addInputResource(root.trim());
            }
            launcher.getEnvironment().setAutoImports(true);
            launcher.getEnvironment().setCommentEnabled(true);
            if (classRoots.length > 0) {
                launcher.getEnvironment().setSourceClasspath(classRoots);
            }
            debugPrintln(">>>> Building spoon model...");
            CtModel model = launcher.buildModel();

            debugPrintln(">>>> Indexing classes...");
            this.classes = (ignoreNestedTypes ? model.getAllTypes().stream() : getAllTopLevelAndNestedTypes(model))
                    .map(e -> {
                        Class c = new Class();
                        if (e instanceof CtInterface) {
                            c.type = "interface";
                        } else if (e instanceof CtEnum) {
                            c.type = "enum";
                        } else {
                            // maybe not class, but do not care
                            c.type = "class";
                        }
                        c.name = e.getSimpleName();
                        c.path = e.getQualifiedName();
                        c.code = getSourcCodeOfElement(e);
                        c.ctType = e;
                        if (e.getPosition().getFile() != null) {
                            c.filePath = e.getPosition().getFile().getAbsolutePath();
                        }
                        return c;
                    }).toList();
            debugPrintln(">>>> Found %d classes", classes.size());

            debugPrintln(">>>> Indexing methods...");
            this.methods = new ArrayList<>();
            this.methods.addAll(model.getElements(new TypeFilter<>(CtMethod.class)).stream().map(e -> {
                Method m = new Method();
                String simpleSignature = getSimpleSignatureOfMethodOrConstructor(e);
                CtType<?> declaringType = e.getDeclaringType();
                String declClassPath = declaringType != null ? declaringType.getQualifiedName() : "unknown";
                m.name = e.getSimpleName();
                m.path = String.format("%s.%s", declClassPath, e.getSimpleName());
                m.classPath = declClassPath;
                m.qualifiedName = String.format("%s.%s", declClassPath, simpleSignature);
                m.code = getSourcCodeOfElement(e);
                m.ctMethod = e;
                if (e.getPosition().getFile() != null) {
                    m.filePath = e.getPosition().getFile().getAbsolutePath();
                    m.startLine = e.getPosition().getLine();
                    m.startColumn = e.getPosition().getColumn();
                    m.endLine = e.getPosition().getEndLine();
                    m.endColumn = e.getPosition().getEndColumn();
                }
                return m;
            }).toList());
            debugPrintln(">>>> Indexing constructors...");
            this.methods.addAll(model.getElements(new TypeFilter<>(CtConstructor.class)).stream().map(e -> {
                Method m = new Method();
                String simpleSignature = getSimpleSignatureOfMethodOrConstructor(e);
                CtType<?> declaringType = e.getDeclaringType();
                String declClassPath = declaringType != null ? declaringType.getQualifiedName() : "unknown";
                // m.name = e.getType()!=null?e.getType().getSimpleName():e.getSimpleName();
                m.name = e.getSimpleName(); // always "<init>"
                m.path = String.format("%s.%s", declClassPath, m.name);
                m.classPath = declClassPath;
                m.qualifiedName = String.format("%s.%s", declClassPath, simpleSignature);
                m.code = getSourcCodeOfElement(e);
                m.ctMethod = e;
                if (e.getPosition().getFile() != null) {
                    m.filePath = e.getPosition().getFile().getAbsolutePath();
                    m.startLine = e.getPosition().getLine();
                    m.startColumn = e.getPosition().getColumn();
                    m.endLine = e.getPosition().getEndLine();
                    m.endColumn = e.getPosition().getEndColumn();
                }
                return m;
            }).toList());
            debugPrintln(">>>> Found %d methods & constructors", methods.size());

            // // Check qualnames
            // {
            // Set<String> names = new HashSet<>();
            // for (Class cls : this.classes) {
            // assert !names.contains(cls.path) : "two classes with one path";
            // names.add(cls.path);
            // }
            // for (Method md : this.methods) {
            // assert !names.contains(md.qualifiedName)
            // : "two methods with one qualname, TODO: fix
            // getSimpleSignatureOfMethodOrConstructor";
            // names.add(md.qualifiedName);
            // }
            // }
        }

        private static String convertWildcardPatternToRegex(String pattern, String style) {
            // Escape special chars in pattern except '*'
            if ("java".equals(style)) {
                // \ ^ $ . | ? + ( ) [ ] { }
                pattern = pattern.replaceAll("[\\\\\\^\\$\\.\\|\\?\\+\\(\\)\\[\\]\\{\\}]", "\\\\$0");
            } else if ("grep".equals(style)) {
                // \ ^ $ . [ ]
                pattern = pattern.replaceAll("[\\\\\\^\\$\\.\\[\\]]", "\\\\$0");
            } else {
                throw new IllegalArgumentException("given style is not valid");
            }
            StringBuilder result = new StringBuilder();
            int length = pattern.length();

            for (int i = 0; i < length; i++) {
                char currentChar = pattern.charAt(i);
                if (currentChar == '\\' && i + 1 < length && pattern.charAt(i + 1) == '*') {
                    result.append("\\*");
                    i++;
                } else if (currentChar == '*') {
                    result.append(".*");
                } else {
                    result.append(currentChar);
                }
            }

            return result.toString();
        }

        // private static String makeNotAllMatchWildcardPattern(String pattern) {
        // String prefix = pattern.startsWith("*") ? "" : "*";
        // String suffix = pattern.endsWith("*") ? "" : "*";
        // return prefix + pattern + suffix;
        // }

        public List<Class> getClasses() {
            return classes;
        }

        public List<Method> getMethods() {
            return methods;
        }

        public List<Class> getClassesInInheritTree(String targetClassPath) {
            Class targetClass = classes.stream()
                    .filter(e -> e.path.equals(targetClassPath))
                    .findFirst().orElse(null);
            if (targetClass == null) {
                debugPrintln("WARN: Target class %s not found", targetClassPath);
                return new ArrayList<>();
            }

            // Collect all subclasses (including subclasses of subclasses, etc.)
            Set<String> subclasses = new HashSet<>();
            subclasses.add(targetClassPath);
            int lastCountOfSubclasses;
            do {
                lastCountOfSubclasses = subclasses.size();
                Set<String> newSubs = classes.stream()
                        .filter(cls -> {
                            String classPath = cls.path;
                            if (subclasses.contains(classPath)) {
                                return false; // Skip already processed classes
                            }

                            CtTypeReference<?> superRef = cls.ctType.getSuperclass();
                            String superClassPath = null;
                            if (superRef != null) {
                                CtType<?> superType = superRef.getTypeDeclaration();
                                if (superType != null) {
                                    superClassPath = superType.getQualifiedName();
                                }
                            }

                            // Include if this class extends one in our current set
                            return superClassPath != null && subclasses.contains(superClassPath);
                        })
                        .map(cls -> cls.path)
                        .collect(Collectors.toSet());

                subclasses.addAll(newSubs);
            } while (subclasses.size() > lastCountOfSubclasses); // Continue until no new subclasses found

            // Collect all superclasses (including superclasses of superclasses, etc.)
            Set<String> superclasses = new HashSet<>();
            CtType<?> currentSuper = targetClass.ctType;
            while (currentSuper != null) {
                superclasses.add(currentSuper.getQualifiedName());
                CtTypeReference<?> superRef = currentSuper.getSuperclass();
                currentSuper = (superRef != null) ? superRef.getTypeDeclaration() : null;
            }

            // Combine both superclasses and subclasses and return
            return classes.stream()
                    .filter(e -> subclasses.contains(e.path) || superclasses.contains(e.path))
                    .collect(Collectors.toList());
        }

        public List<Class> searchClass(String wildcardPattern) {
            // wildcardPattern match the class qualified name (Class.path)
            final String wildcardPattern0 = convertWildcardPatternToRegex(wildcardPattern.replace("\\s", ""), "java");
            return classes.stream().filter(e -> {
                return e.filePath != null && e.path.matches(wildcardPattern0);
            }).toList();
        }

        public List<Method> searchMethod(String wildcardPattern) {
            // wildcardPattern match the method path + parameter types
            // (Method.qualifiedName)
            final String wildcardPattern0 = convertWildcardPatternToRegex(wildcardPattern.replace("\\s", ""), "java");
            return methods.stream().filter(e -> {
                return e.filePath != null && e.qualifiedName.matches(wildcardPattern0);
            }).toList();
        }

        public List<CodeSnippet> searchCode(String wildcardPattern, int numContextLines) {
            // Prepare the grep command
            final String BLOCK_SEP = "__block_sep__";
            List<String> commandList = new ArrayList<>();
            commandList.add("grep");
            commandList.add("-r");
            commandList.add("-C");
            commandList.add(String.valueOf(numContextLines));
            commandList.add("-n");
            commandList.add("--include=*.java");
            commandList.add("--color=never");
            commandList.add("--group-separator=" + BLOCK_SEP);
            commandList.add(convertWildcardPatternToRegex(wildcardPattern, "grep"));
            for (int i = 0; i < sourceRoots.length; i++) {
                commandList.add(sourceRoots[i]);
            }
            String[] command = commandList.toArray(new String[0]);

            // Launch the command and parse the output
            List<CodeSnippet> snippets = new ArrayList<>();
            try {
                Process process = Runtime.getRuntime().exec(command);
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                    String line;
                    CodeSnippet currentSnippet = new CodeSnippet();
                    while ((line = reader.readLine()) != null) {
                        if (BLOCK_SEP.equals(line.strip())) {
                            // end of a snippet
                            assert currentSnippet.lines.isEmpty();
                            snippets.add(currentSnippet);
                            currentSnippet = new CodeSnippet();
                        } else {
                            // parse the line
                            Pattern pattern = Pattern.compile("^(.*?)[:-](\\d+)[:-](.*)$");
                            Matcher matcher = pattern.matcher(line);
                            if (matcher.find()) {
                                String fileName = matcher.group(1);
                                int lineNumber = Integer.parseInt(matcher.group(2));
                                String content = matcher.group(3);

                                if (currentSnippet.filePath == null) {
                                    currentSnippet.filePath = fileName;
                                } else {
                                    assert currentSnippet.filePath.equals(fileName);
                                }
                                currentSnippet.lineNumbers.add(lineNumber);
                                currentSnippet.lines.add(content);
                            }
                        }

                    }
                    if (!currentSnippet.lines.isEmpty()) {
                        snippets.add(currentSnippet);
                    }
                    currentSnippet = null;
                }
                process.waitFor();
            } catch (IOException | InterruptedException e) {
                throw new RuntimeException(e);
            }

            // Fill-in other fields of the snippets
            for (var snippet : snippets) {
                int startLine = snippet.lineNumbers.get(0);
                int endLine = snippet.lineNumbers.get(snippet.lineNumbers.size() - 1);
                try {
                    String[] pathTyAndPath = getPkgClsMdPathByPosition(snippet.filePath, startLine, endLine);
                    snippet.pathType = pathTyAndPath[0];
                    snippet.path = pathTyAndPath[1];
                } catch (IllegalArgumentException ex) {
                    snippet.pathType = "unknown";
                    snippet.path = "unknown";
                }
                snippet.code = String.join("\n", snippet.lines);
            }

            return snippets;
        }
    }
}
