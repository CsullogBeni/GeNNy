package p4query.experts.syntaxtree.tasksolutions;

import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;
import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.__;
import org.apache.tinkerpop.gremlin.structure.Graph;
import org.apache.tinkerpop.gremlin.tinkergraph.structure.TinkerGraph;
import org.apache.tinkerpop.gremlin.structure.io.graphson.GraphSONIo;

import java.io.IOException;
import java.util.List;
import java.util.Set;

import static org.apache.tinkerpop.gremlin.process.traversal.P.within;

import p4query.experts.syntaxtree.tasksolutions.FirstTask;

public class TaskSolutions {

    public static void main(String[] args) {
        GraphTraversalSource g = load_graph();

        // FirstTask firstTask = new FirstTask(g, "ethernet", "dstAddr");
        // firstTask.run();

        // SecondTask secondTask = new SecondTask(g, "ipv4", "isValid");
        // secondTask.run();

        //ThirdTask thirdTask = new ThirdTask(g);
        //thirdTask.run();

        FourthTask fourthTask = new FourthTask(g, "TYPE_IPV4", "default");
        fourthTask.run();


    }

    private static GraphTraversalSource load_graph() {
        Graph graph = TinkerGraph.open();

        try {
            graph.io(GraphSONIo.build()).readGraph("C:\\Git\\ir\\experts-syntax-tree\\src\\main\\java\\p4query\\experts\\syntaxtree\\tasksolutions\\graph.json");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        GraphTraversalSource g = graph.traversal();

        // g.V().forEachRemaining(System.out::println);

        return g;
    }


    private void third_task(GraphTraversalSource g) {
        System.out.println("=============================third_task==============================================");

        List<Object> ipv4_forwards = g.V()
                .has("class", "ActionDeclarationContext")
                .repeat(__.out())
                .until(__.not(__.out()))
                .has("value", within("ipv4_forward"))
                .values("nodeId")
                .toList();

        //ipv4_forwards.forEach(System.out::println);

        //"TerminalNodeImpl"
        Set<Object> ipv4_forwards_parent = g.V()
                .has("class", "ActionDeclarationContext")
                .where(
                        __.repeat(__.out()).until(__.not(__.out())).has("nodeId", within(ipv4_forwards))
                )
                .values("nodeId")
                .toSet();

        //ipv4_forwards_parent.forEach(System.out::println);

        List<Object> ipv4_fields = g.V()
                .has("class", "TerminalNodeImpl")
                .repeat(
                        __.in().has("nodeId", within(ipv4_forwards_parent))
                )
                .values("value")
                .toList();

        ipv4_fields.forEach(System.out::println);

        System.out.println("=====================================================================================");
    }


}
