package comp5216.sydney.edu.au.todolist;

/**
 * Created by zhlu2106 on 23/08/2016.
 */
public class Item {
    public String todo;
    public String creation_time;
    public String photo_path;
    public String location;

    public Item(String ToDo, String creation_time, String photo_path, String location) {
        this.todo = ToDo;
        this.creation_time = creation_time;
        this.photo_path = photo_path;
        this.location = location;
    }
}
