package comp5216.sydney.edu.au.todolist;

import com.orm.SugarRecord;

public class ToDoItem extends SugarRecord {
    public String todo;
    public String time;
    public String photo_path;
    public String location;

    public ToDoItem(){}

    public ToDoItem(String ToDo){
        this.todo = ToDo;
    }

    public ToDoItem(String ToDo, String creation_time, String photo_path, String location) {
        this.todo = ToDo;
        this.time = creation_time;
        this.photo_path = photo_path;
        this.location = location;
    }
}
