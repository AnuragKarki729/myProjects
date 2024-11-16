import React from 'react';
import { Outlet, Link } from 'react-router-dom';
import { Navbar, Nav, Container } from 'react-bootstrap';  

function Root() {
    return (
        <div>
        <Navbar bg="dark" variant="dark" expand="lg">
            <Container>
            <Navbar.Brand>Carada</Navbar.Brand>
            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav">
                <Nav className="mb-auto">
                <Nav.Link as={Link} to="/WAD_Project/">Dashboard</Nav.Link>
                <Nav.Link as={Link} to="/WAD_Project/Car-Details">Car Details Page</Nav.Link>
                </Nav>
            </Navbar.Collapse>
            </Container>
        </Navbar>

        <Outlet />
        </div>
    );
}

export default Root;
