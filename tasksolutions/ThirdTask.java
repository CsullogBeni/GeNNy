package p4query.experts.syntaxtree.tasksolutions;

import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.__;

import java.util.List;

import static org.apache.tinkerpop.gremlin.process.traversal.P.within;

public class ThirdTask implements Task {
    GraphTraversalSource g;
    String header;
    String field;
    List<Object> lines;

    public ThirdTask(GraphTraversalSource g/*, String header, String field*/) {
        this.g = g;
        /* this.header = header;
        this.field = field; */
    }

    public void run() {
        System.out.println("=============================third_task==============================================");
        this.lines = query();
        lines.forEach(System.out::println);
        System.out.println("=====================================================================================");
    }

    public List<Object> query() {
        System.out.println("get_ipv4_forward_action_declaration: " + get_ipv4_forward_action_declaration());
        List<Object> ipv4_forward_action_declaration = get_ipv4_forward_action_declaration();
        return g.V()
                .has("nodeId", within(ipv4_forward_action_declaration))
                .repeat(__.out()).until(
                        __.and(
                                __.has("class", "AssignmentOrMethodCallStatementContext"),
                                __.repeat(__.out()).until(
                                        __.and(
                                                __.has("class", "LvalueContext"),
                                                __.repeat(__.out()).until(
                                                        __.and(
                                                                __.has("class", "TerminalNodeImpl")
                                                        )
                                                )
                                        )
                                )
                        )
                )
                .values("nodeId")
                .toList();
    }

    private List<Object> get_ipv4_forward_action_declaration() {
        return  g.V()
                .has("value", "ipv4_forward")
                .has("class", "TerminalNodeImpl")
                .repeat(__.in()).until(
                        __.and(
                                __.has("class", "ActionDeclarationContext")
                        )
                )
                .values("nodeId")
                .toList();
    }
}
