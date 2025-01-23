package p4query.experts.syntaxtree.tasksolutions;

import org.apache.tinkerpop.gremlin.process.traversal.dsl.graph.GraphTraversalSource;

import java.util.List;

public interface Task {
    void run();

    private List<Object> query(){
        return null;
    }
}
