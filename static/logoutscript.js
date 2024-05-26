document.addEventListener('DOMContentLoaded', function () {
    const logoutButton = document.getElementById('logoutButton');

    logoutButton.addEventListener('click', function () {
        // Redirect to the logout route
        window.location.href = "/logout";
    });
});