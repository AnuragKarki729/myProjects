import { useRouteError } from "react-router-dom";

import 'bootstrap/dist/css/bootstrap.min.css';
import { Container } from "react-bootstrap";


export default function ErrorPage() {
    const error = useRouteError();
    console.error(error);

    return (
        <div>
            <Container className="my-5">
                <h1>Oops!</h1>
                <p>Sorry, an unexpected error has occurred.</p>
                <p>
                    <i>{error?.statusText || error?.message || "Unknown error"}</i>
                </p>
            </Container>
        </div>
    );
};