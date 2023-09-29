def get_styles():
    return """
    <style>
        .navbar-wrapper-3 {
            width: auto;
            justify-content: space-between;
            align-items: center;
            display: flex;
        }

        body {
            color: #333;
            font-family: Arial, Helvetica Neue, Helvetica, sans-serif;
            font-size: 14px;
            line-height: 20px;
            min-height: 100%;
            background-color: #fff;
            margin: 0;
        }

        html {
            -ms-text-size-adjust: 100%;
            -webkit-text-size-adjust: 100%;
            font-family: sans-serif;
        }

        /* Add this to change the color of the navbar links */
        .navbar-dark .navbar-nav .nav-link {
            color: #671e75;
        }

        /* If you want to change the color on hover as well */
        .navbar-dark .navbar-nav .nav-link:hover {
            color: #501260;  /* This is a slightly darker shade, adjust as needed */
        }
    </style>
    """
