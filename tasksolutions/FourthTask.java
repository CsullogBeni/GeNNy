package p4query.experts.syntaxtree.tasksolutions;

import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.__;

import java.util.List;

public class FourthTask implements Task {

    GraphTraversalSource g;
    String header;
    String field;
    List<Object> lines;

    public FourthTask(GraphTraversalSource g, String header, String field) {
        this.g = g;
        this.header = header;
        this.field = field;
    }

    public void run() {
        System.out.println("=============================fourth_task=============================================");
        this.lines = query();
        // lines.forEach(System.out::println);
        System.out.println("State:");
        get_state().forEach(System.out::println);
        System.out.println("=====================================================================================");
    }

    private List<Object> query() {
        return g.V()
                .has("class", "SelectExpressionContext")
                .repeat(__.out()).until(
                        __.or(
                                __.and(
                                        __.has("class", "TerminalNodeImpl"),
                                        __.has("value", header),
                                        __.repeat(__.in()).until(
                                                __.and(
                                                        __.has("class", "SimpleKeysetExpressionContext")
                                                )
                                        )
                                ),
                                __.and(
                                        __.has("class", "TerminalNodeImpl"),
                                        __.has("value", field),
                                        __.repeat(__.in()).until(
                                                __.and(
                                                        __.has("class", "SimpleKeysetExpressionContext")
                                                )
                                        )
                                )
                        )
                )
                .values("nodeId")
                .toList();
    }

    private List<Object> get_state() {
        return g.V()
                .has("class", "ParserStatesContext")
                .repeat(__.out()).until(
                        __.and(
                                __.has("class", "TerminalNodeImpl"),
                                __.and(
                                        __.repeat(__.in()).until(
                                                __.and(

                                                        __.has("class", "ParserStateContext"),
                                                        __.repeat(__.out()).until(
                                                                __.and(
                                                                        __.has("nodeId", lines.get(0))
                                                                )
                                                        )
                                                )
                                        ),
                                        __.repeat(__.in()).until(
                                                __.and(
                                                        __.has("class", "Type_or_idContext"),
                                                        __.repeat(__.in()).until(
                                                                __.and(
                                                                        __.has("class", "NonTypeNameContext"),
                                                                        __.repeat(__.in()).until(
                                                                                __.and(
                                                                                        __.has("class", "NameContext")
                                                                                )
                                                                        )
                                                                )
                                                        )
                                                )
                                        )

                                )

                        )

                )
                .values("value")
                .toList();
    }
}

