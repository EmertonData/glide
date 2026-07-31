(function () {
  const container = document.getElementById("quizz-app");
  const PASS_RATIO = 0.8;

  function renderIntro(total) {
    const intro = document.createElement("p");
    intro.className = "quizz-intro";
    const passScore = Math.ceil(PASS_RATIO * total);
    intro.textContent =
      "This quiz has " + total + " questions. A score of at least " +
      passScore + "/" + total + " (" + Math.round(PASS_RATIO * 100) +
      "%) is required to pass.";
    container.appendChild(intro);
  }

  function renderQuestions(questions) {
    const form = document.createElement("form");
    form.id = "quizz-form";

    questions.forEach(function (question, index) {
      const fieldset = document.createElement("fieldset");
      fieldset.className = "quizz-question";

      const legend = document.createElement("legend");
      legend.textContent = (index + 1) + ". " + question.question;
      fieldset.appendChild(legend);

      question.options.forEach(function (option, optionIndex) {
        const label = document.createElement("label");
        label.className = "quizz-option";

        const input = document.createElement("input");
        input.type = "radio";
        input.name = "q" + index;
        input.value = optionIndex;

        label.appendChild(input);
        label.appendChild(document.createTextNode(option));
        fieldset.appendChild(label);
      });

      const feedback = document.createElement("p");
      feedback.className = "quizz-feedback";
      feedback.id = "feedback-" + index;
      fieldset.appendChild(feedback);

      form.appendChild(fieldset);
    });

    const submitButton = document.createElement("button");
    submitButton.type = "submit";
    submitButton.textContent = "Submit";
    submitButton.className = "quizz-submit";
    form.appendChild(submitButton);

    const result = document.createElement("div");
    result.id = "quizz-result";
    result.className = "quizz-result";
    form.appendChild(result);

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      gradeQuizz(questions);
    });

    container.appendChild(form);
  }

  function gradeQuizz(questions) {
    let score = 0;

    questions.forEach(function (question, index) {
      const selected = document.querySelector('input[name="q' + index + '"]:checked');
      const feedback = document.getElementById("feedback-" + index);
      const selectedIndex = selected ? Number(selected.value) : null;
      const isCorrect = selectedIndex === question.answerIndex;

      if (isCorrect) {
        score += 1;
      }

      feedback.textContent = isCorrect
        ? "Correct. " + question.explanation
        : 'Incorrect. Correct answer: "' + question.options[question.answerIndex] + '". ' + question.explanation;
      feedback.className = "quizz-feedback " + (isCorrect ? "quizz-correct" : "quizz-incorrect");
    });

    const total = questions.length;
    const passScore = Math.ceil(PASS_RATIO * total);
    const passed = score >= passScore;
    const percentage = Math.round((score / total) * 100);

    const result = document.getElementById("quizz-result");
    result.textContent =
      "Score: " + score + "/" + total + " (" + percentage + "%) — " +
      (passed ? "Pass" : "Not yet, review the feedback above and retake the quiz");
    result.className = "quizz-result " + (passed ? "quizz-pass" : "quizz-fail");
    result.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  fetch(QUIZZ_SOURCE)
    .then(function (response) {
      if (!response.ok) {
        throw new Error("Failed to load " + QUIZZ_SOURCE + ": " + response.status);
      }
      return response.json();
    })
    .then(function (questions) {
      renderIntro(questions.length);
      renderQuestions(questions);
    })
    .catch(function (error) {
      container.textContent =
        "Could not load quiz questions (" + error.message + "). If you opened this file directly " +
        "(file://), serve this folder over HTTP instead — see the loading instructions in the " +
        "onboarding checklist.";
    });
})();
