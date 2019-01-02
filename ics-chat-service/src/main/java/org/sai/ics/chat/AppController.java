package org.sai.ics.chat;

import org.springframework.web.bind.annotation.*;

@RestController
public class AppController {

    @PostMapping("/input")
    public String botResponse(@RequestBody String input) {
        System.out.println("Reached controller");
        String botMessage = input;

        if (input == "hi" || input =="hello") {
            String[] hi = {"Hi", "Howdy", "Hello", "Greetings"};
            botMessage = hi[(int) Math.floor((Math.random() * hi.length))];
        }
        if (input.contains("name")) {
            botMessage = "My name is Chatty";
        }

        return botMessage;
    }
}
