package org.sai.ics.chat;

import org.springframework.web.bind.annotation.*;
import java.io.*;

@RestController
public class AppController {

    @PostMapping("/input")
    public String botResponse(@RequestBody String input) {
        String botMessage = "";

        if (input.equals("hi") || input.equals("hello")) {
            String[] hi = {"Hi", "Howdy", "Hello", "Greetings"};
            botMessage = hi[(int) Math.floor((Math.random() * hi.length))];
        } else if (input.contains("name")) {
            botMessage = "My name is Chatty";
        } else {
            try {
                String[] cmd = new String[]{"python", "./ics-chat-model/main.py", input};
                Process p = Runtime.getRuntime().exec(cmd);
                BufferedReader stdInput = new BufferedReader(new
                        InputStreamReader(p.getInputStream()));
                botMessage = stdInput.readLine();
            } catch (IOException e) {
                System.out.println("Exception");
                e.printStackTrace();
                System.exit(-1);
            }
        }

        return botMessage;
    }
}
