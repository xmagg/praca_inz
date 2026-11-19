"use strict";

(() => {
    const form = document.querySelector('form[action="/predict"]');
    if (!form) {
        return;
    }

    const fields = Array.from(form.querySelectorAll("input, select, textarea"));
    let focusPending = false;

    function validationMessage(field) {
        const validity = field.validity;

        if (validity.badInput || validity.typeMismatch) {
            return "Wpisz poprawną liczbę.";
        }

        if (validity.valueMissing) {
            return field.tagName === "SELECT"
                ? "Wybierz wartość."
                : "Uzupełnij to pole.";
        }

        if (validity.rangeUnderflow) {
            return `Minimalna dopuszczalna wartość to ${field.min}.`;
        }

        if (validity.rangeOverflow) {
            return `Maksymalna dopuszczalna wartość to ${field.max}.`;
        }

        if (validity.stepMismatch) {
            return !field.step || field.step === "1"
                ? "Wpisz liczbę całkowitą."
                : `Wpisz wartość z krokiem ${field.step.replace(".", ",")}.`;
        }

        return field.validationMessage || "Popraw wartość w tym polu.";
    }

    function showError(field) {
        const errorId = `${field.id}-error`;
        let error = document.getElementById(errorId);

        if (!error) {
            error = document.createElement("p");
            error.id = errorId;
            error.className = "field-error";
            error.setAttribute("aria-live", "polite");
            field.insertAdjacentElement("afterend", error);
        }

        error.textContent = validationMessage(field);
        field.setAttribute("aria-invalid", "true");

        const descriptions = (field.getAttribute("aria-describedby") || "")
            .split(/\s+/)
            .filter(Boolean);

        if (!descriptions.includes(errorId)) {
            descriptions.push(errorId);
        }

        field.setAttribute("aria-describedby", descriptions.join(" "));
    }

    function clearError(field) {
        const errorId = `${field.id}-error`;

        document.getElementById(errorId)?.remove();
        field.removeAttribute("aria-invalid");

        const descriptions = (field.getAttribute("aria-describedby") || "")
            .split(/\s+/)
            .filter((id) => id && id !== errorId);

        if (descriptions.length) {
            field.setAttribute("aria-describedby", descriptions.join(" "));
        } else {
            field.removeAttribute("aria-describedby");
        }
    }

    form.addEventListener("invalid", (event) => {
        const field = event.target;

        if (!fields.includes(field)) {
            return;
        }

        event.preventDefault();
        showError(field);

        if (!focusPending) {
            focusPending = true;

            requestAnimationFrame(() => {
                focusPending = false;

                const firstInvalid = fields.find(
                    (item) => !item.validity.valid
                );

                if (firstInvalid) {
                    firstInvalid.focus({ preventScroll: true });

                    firstInvalid.scrollIntoView({
                        behavior: "instant",
                        block: "nearest"
                    });
                }
            });
        }
    }, true);

    fields.forEach((field) => {
        function refreshError() {
            if (field.getAttribute("aria-invalid") !== "true") {
                return;
            }

            if (field.validity.valid) {
                clearError(field);
            } else {
                showError(field);
            }
        }

        field.addEventListener("input", refreshError);
        field.addEventListener("change", refreshError);
    });

    form.addEventListener("reset", () => fields.forEach(clearError));
})();