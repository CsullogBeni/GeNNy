package p4query.experts.syntaxtree.tasksolutions;

import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.__;
import p4query.experts.syntaxtree.p4.P4Parser;

import java.util.LinkedList;
import java.util.List;

public class SecondTask implements Task {
    GraphTraversalSource g;
    String header;
    String field;
    List<Object> lines;
    List<Object> control_name_lines;
    List<Object> control_name;

    public SecondTask(GraphTraversalSource g, String header, String field) {
        this.g = g;
        this.header = header;
        this.field = field;
    }

    public void run() {
        System.out.println("=============================second_task=============================================");
        this.lines = query();
        lines.forEach(System.out::println);
        search_control_names();
        // control_name_lines.forEach(System.out::println);
        System.out.println("=====================================================================================");
    }

    private List<Object> query() {
        return g.V()
                .has("class", "StatementOrDeclarationContext")
                .repeat(__.out()).until(
                        __.and(
                                __.has("class", "ConditionalStatementContext"),
                                __.and(
                                        __.out().and(
                                                __.has("class", "TerminalNodeImpl"),
                                                __.has("value", "if")
                                        ),
                                        __.out().and(
                                                __.has("class", "TerminalNodeImpl"),
                                                __.has("value", "(")
                                        ),
                                        __.out().and(
                                                __.has("class", "TerminalNodeImpl"),
                                                __.has("value", ")")
                                        ),
                                        __.out().and(
                                                __.has("class", "ExpressionContext"),
                                                __.repeat(__.out()).until(
                                                        __.or(
                                                                __.and(
                                                                        __.has("class", "TerminalNodeImpl"),
                                                                        __.has("value", ")")
                                                                ),
                                                                __.and(
                                                                        __.has("class", "TerminalNodeImpl"),
                                                                        __.has("value", "(")
                                                                ),
                                                                __.and(
                                                                        __.has("class", "ExpressionContext"),
                                                                        __.repeat(__.out()).until(
                                                                                __.or(
                                                                                        __.and(
                                                                                                __.has("class", "ExpressionContext"),
                                                                                                __.repeat(__.out()).until(
                                                                                                        __.and(
                                                                                                                __.and(
                                                                                                                        __.has("class", "ExpressionContext"),
                                                                                                                        __.repeat(__.out()).until(
                                                                                                                                __.and(
                                                                                                                                        __.has("class", "TerminalNodeImpl"),
                                                                                                                                        __.has("value", "hdr")
                                                                                                                                )
                                                                                                                        )
                                                                                                                ),
                                                                                                                __.and(
                                                                                                                        __.has("class", "NameContext"),
                                                                                                                        __.repeat(__.out()).until(
                                                                                                                                __.and(
                                                                                                                                        __.has("class", "TerminalNodeImpl"),
                                                                                                                                        __.has("value", header)
                                                                                                                                )
                                                                                                                        )
                                                                                                                )
                                                                                                        )
                                                                                                )
                                                                                        ),
                                                                                        __.and(
                                                                                                __.has("class", "NameContext"),
                                                                                                __.repeat(__.out()).until(
                                                                                                        __.and(
                                                                                                                __.has("class", "TerminalNodeImpl"),
                                                                                                                __.has("value", field)
                                                                                                        )
                                                                                                )
                                                                                        )

                                                                                )
                                                                        )
                                                                )
                                                        )
                                                )
                                        )
                                )
                        )
                )
                .values("line")
                .toList();
    }

    private void search_control_names() {
        for (Object line : lines) {
            /*control_name_lines = g.V()
                    .has("class", "ControlDeclarationContext")
                    .and(
                            __.out().and(
                                    __.has("class", "ControlTypeDeclarationContext"),
                                    __.and(
                                            __.out().and(
                                                    __.has("class", "NameContext"),
                                                    __.repeat(__.out()).until(
                                                            __.has("class", "TerminalNodeImpl")
                                                    )
                                            )
                                    )
                            )
                    )
                    .values("line")
                    .toList();*/


            control_name = g.V()
                    .has("class", "TerminalNodeImpl")
                    .and(
                            __.repeat(__.in()).until(
                                    __.and(
                                            __.has("class", "NameContext"),
                                            __.and(

                                                    __.repeat(__.in()).until(
                                                            __.and(
                                                                    __.has("class", "ControlDeclarationContext"),
                                                                    __.repeat(__.out()).until(
                                                                            __.and(
                                                                                    __.has("class", "TerminalNodeImpl"),
                                                                                    __.has("value", "if"),
                                                                                    __.has("line", line)
                                                                            )
                                                                    )
                                                            )
                                                    ), __.repeat(__.in()).until(
                                                            __.and(
                                                                    __.has("class", "ControlTypeDeclarationContext").out().has("value", "control")
                                                            )
                                                    )
                                            )
                                    )
                            )
                    )
                    .values("value")
                    .toList();

            control_name.forEach(System.out::println);


            //control_name_lines.forEach(System.out::println);
        }
    }
}
