package p4query.experts.syntaxtree.tasksolutions;


import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.__;

import java.util.LinkedList;
import java.util.List;

public class FirstTask implements Task {

    GraphTraversalSource g;
    String header;
    String field;
    List<Object> lines;
    List<Object> contents;

    public FirstTask(GraphTraversalSource g, String header, String field) {
        this.g = g;
        this.header = header;
        this.field = field;
    }

    public void run() {

        System.out.println("=============================first_task==============================================");
        this.lines = query();
        this.contents = new LinkedList<>();
        get_assignment_content();
        System.out.println("Lines that contain hdr." + header + "." + field + ":");
        lines.forEach(System.out::println);
        System.out.println("Contents:");
        contents.forEach(System.out::println);
        System.out.println("=====================================================================================");
    }

    private List<Object> query() {
        return g.V()
                .has("class", "StatementOrDeclarationContext")
                .repeat(__.out()).until(
                        __.and(
                                __.has("class", "AssignmentOrMethodCallStatementContext"),
                                __.repeat(__.out()).until(
                                        __.or(
                                                __.and(
                                                        __.has("class", "LvalueContext"),
                                                        __.repeat(__.out()).until(
                                                                __.and(
                                                                        __.has("class", "LvalueContext"),
                                                                        __.repeat(__.out()).until(
                                                                                __.and(
                                                                                        __.has("class", "PrefixedNonTypeNameContext"),
                                                                                        __.repeat(__.out()).until(
                                                                                                __.and(
                                                                                                        __.has("value", "hdr"),
                                                                                                        __.has("class", "TerminalNodeImpl")
                                                                                                )
                                                                                        )
                                                                                )
                                                                        ),
                                                                        __.repeat(__.out()).until(
                                                                                __.and(
                                                                                        __.has("class", "NonTypeNameContext"),
                                                                                        __.repeat(__.out()).until(
                                                                                                __.and(
                                                                                                        __.has("value", header),
                                                                                                        __.has("class", "TerminalNodeImpl")

                                                                                                )
                                                                                        )
                                                                                )
                                                                        )
                                                                )
                                                        ),
                                                        __.repeat(__.out()).until(
                                                                __.and(
                                                                        __.has("class", "NonTypeNameContext"),
                                                                        __.repeat(__.out()).until(
                                                                                __.and(
                                                                                        __.has("value", field),
                                                                                        __.has("class", "TerminalNodeImpl")

                                                                                )
                                                                        )
                                                                )
                                                        ),
                                                        __.repeat(__.out()).until(
                                                                __.and(
                                                                        __.has("value", "."),
                                                                        __.has("class", "TerminalNodeImpl")
                                                                )
                                                        )
                                                ),
                                                __.and(
                                                        __.has("class", "ExpressionContext"),
                                                        __.repeat(__.out()).until(
                                                                __.and(
                                                                        __.has("class", "ExpressionContext"),
                                                                        __.repeat(__.out()).until(
                                                                                __.and(
                                                                                        __.has("class", "NonTypeNameContext"),
                                                                                        __.repeat(__.out()).until(
                                                                                                __.and(
                                                                                                        __.has("value", "hdr"),
                                                                                                        __.has("class", "TerminalNodeImpl")
                                                                                                )
                                                                                        )
                                                                                )
                                                                        ),
                                                                        __.repeat(__.out()).until(
                                                                                __.and(
                                                                                        __.has("class", "NonTypeNameContext"),
                                                                                        __.repeat(__.out()).until(
                                                                                                __.and(
                                                                                                        __.has("value", header),
                                                                                                        __.has("class", "TerminalNodeImpl")

                                                                                                )
                                                                                        )
                                                                                )
                                                                        )
                                                                )
                                                        ),
                                                        __.repeat(__.out()).until(
                                                                __.and(
                                                                        __.has("class", "NonTypeNameContext"),
                                                                        __.repeat(__.out()).until(
                                                                                __.and(
                                                                                        __.has("value", field),
                                                                                        __.has("class", "TerminalNodeImpl")

                                                                                )
                                                                        )
                                                                )
                                                        ),
                                                        __.repeat(__.out()).until(
                                                                __.and(
                                                                        __.has("value", "."),
                                                                        __.has("class", "TerminalNodeImpl")
                                                                )
                                                        )
                                                )
                                        )
                                ),
                                __.repeat(__.out()).until(
                                        __.and(
                                                __.has("value", "="),
                                                __.has("class", "TerminalNodeImpl")
                                        )
                                )
                        )
                )
                .values("line")
                .toList();
    }

    public void get_assignment_content() {

        for (Object line : lines) {
            List<Object> new_line = g.V()
                    .repeat(__.out()).until(
                            __.and(
                                    __.has("class", "TerminalNodeImpl"),
                                    __.has("line", line)
                            )
                    )
                    .dedup()
                    .order().by("nodeId")
                    .values("value")
                    .toList();
            String new_content = "";
            boolean first = true;
            for (Object s : new_line) {
                if (first) {
                    new_content = new_content + s;
                    first = false;
                } else {
                    new_content = new_content + " " + s.toString();
                }
            }
            new_content = new_content.replace(" . ", ".");
            new_content = new_content.replace(" ;", ";");
            contents.add(new_content);
        }
    }
}
