// src/pages/CarDetailsPage.jsx
import React, { useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { Form, Button, Container, Row, Col, Card,} from 'react-bootstrap';
import debounce from 'lodash.debounce';

const CarDetailsPage = () => {
    const { mkID } = useParams();
    const [cars, setCars] = useState([]);
    const [brand, setBrand] = useState([]);
    const [highlight, setHighlight] = useState(() => {
        const saved = localStorage.getItem('highlight');
        return saved ? JSON.parse(saved) : [];
    });
    const [highlightedCars, setHighlightedCars] = useState([]);
    const [filteredCars, setFilteredCars] = useState([]);
    const [show, setShow] = useState([...highlight]);
    const [filters, setFilters] = useState({
        brand: '',
        year: '',
        province: '',
        status: ''
    });
    const [sortOrder, setSortOrder] = useState('price');
    
    useEffect(() => {
        fetch('./cars.json')
            .then(response => response.json())
            .then(data => {
                const brandMap = data.MMList.reduce((map, brand) => {
                    map[brand.mkID] = brand.Name;
                    return map;
                }, {});
                
                const carsWithBrands = data.Cars.map(car => ({
                    ...car,
                    Brand: brandMap[car.MkID] || 'Unknown'
                }));
    
                setCars(carsWithBrands);
                setFilteredCars(carsWithBrands);
                
                setBrand([...new Set(carsWithBrands.map(car => car.Brand))]);
            });
    }, []);
    
    useEffect(() => {
        localStorage.setItem('highlight', JSON.stringify(highlight));
    }, [highlight]);
    
    const debouncedFilterAndSort = useCallback(debounce((cars, filters, sortOrder) => {
        let result = cars;
    
        if (filters.brand) {
            result = result.filter(car => car.Brand === filters.brand);
        }
        if (filters.year) {
            result = result.filter(car => car.Yr === parseInt(filters.year));
        }
        if (filters.province) {
            result = result.filter(car => car.Province === filters.province);
        }
        if (filters.status) {
            result = result.filter(car => car.Status === filters.status);
        }
    
        if (sortOrder === 'recent') {
            result.sort((a, b) => b.Yr - a.Yr);
        } else if (sortOrder === 'price') {
            result.sort((a, b) => {
                const priceA = parseInt(a.Prc.replace(/[^0-9]/g, ''));
                const priceB = parseInt(b.Prc.replace(/[^0-9]/g, ''));
                return priceA - priceB;
            });
        }
    
        setFilteredCars(result);
    }, 300), []);
    
    useEffect(() => {
        debouncedFilterAndSort(cars, filters, sortOrder);
    }, [cars, filters, sortOrder, debouncedFilterAndSort]);
    
    const handleFilterChange = (e) => {
        const { name, value } = e.target;
        setFilters(prevFilters => ({
            ...prevFilters,
            [name]: value
        }));
    };
    
    const handleSort = (order) => {
        setSortOrder(order);
    };
    
    const clear = () => {
        setHighlight([]);
    };
    
    const removeHighlight = (car) => {
        setHighlight(prevHighlight => prevHighlight.filter(item => item.Cid !== car.Cid));
    };
    
    const highlightCar = (car) => {
        setHighlight(prevHighlight => {
            if (prevHighlight.some(c => c.Cid === car.Cid)){
                const high = prevHighlight.filter(c => c.Cid !== car.Cid);
                return [...high, car];
            } else {
                return [...prevHighlight, car];
            }
        });
    };
    
    const noResultsMessage = () => {
        const { year, province, status } = filters;
        const filterDescriptions = [
            year && `year ${year}`,
            province && `province ${province}`,
            status && `status ${status}`
        ].filter(Boolean).join(', ');
    
        return `No cars found with ${filterDescriptions}`;
    };
    
    return (
        <div>
            <Container className="my-4">
                <h2 className="my-4">All Cars</h2>
    
                {highlight.length > 0 && (
                    <div className="highlighted-cars mb-4">
                        <h3>Highlighted Cars</h3>
                        <Button variant="danger" onClick={clear} className="mb-3">
                            Clear All
                        </Button>
                        <Row>
                            {highlight.map(car => (
                                <Col xs={12} sm={6} md={4} lg={3} key={car.Cid} className="mb-4">
                                    <Card>
                                        <Card.Img variant="top" src={car.Img300} />
                                        <Card.Body>
                                            <Card.Title>{car.NameMMT}</Card.Title>
                                            <Card.Text>
                                                <strong>Price:</strong> {car.Prc}<br />
                                                <strong>Year:</strong> {car.Yr}<br />
                                                <strong>Status:</strong> {car.Status}<br />
                                                <strong>Brand:</strong> {car.Brand}
                                            </Card.Text>
                                            <Button variant="danger" onClick={() => removeHighlight(car)}>
                                                Remove Highlight
                                            </Button>
                                        </Card.Body>
                                    </Card>
                                </Col>
                            ))}
                        </Row>
                    </div>
                )}
    
                <div className="filters mb-4">
                    <Row className="align-items-center">
                        <Col xs={12} sm={6} md={3}>
                            <Form.Group controlId="formBrand">
                                <Form.Label>Brand:</Form.Label>
                                <Form.Select name="brand" value={filters.brand} onChange={handleFilterChange}>
                                    <option value="">Any Brand</option>
                                    {[...new Set(cars.map(car => car.Brand))].map(brand => (
                                        <option key={brand} value={brand}>{brand}</option>
                                    ))}
                                </Form.Select>
                            </Form.Group>
                        </Col>

                        <Col xs={12} sm={6} md={3}>
                            <Form.Group controlId="formYear">
                                <Form.Label>Year:</Form.Label>
                                <Form.Select name="year" value={filters.year} onChange={handleFilterChange}>
                                    <option value="">Any Year</option>
                                    {[...new Set(cars.map(car => car.Yr))].map(year => (
                                        <option key={year} value={year}>{year}</option>
                                    ))}
                                </Form.Select>
                            </Form.Group>
                        </Col>

                        <Col xs={12} sm={6} md={3}>
                            <Form.Group controlId="formProvince">
                                <Form.Label>Province:</Form.Label>
                                <Form.Select name="province" value={filters.province} onChange={handleFilterChange}>
                                    <option value="">All Province</option>
                                    {[...new Set(cars.map(car => car.Province))].map(province => (
                                        <option key={province} value={province}>{province}</option>
                                    ))}
                                </Form.Select>
                            </Form.Group>
                        </Col>

                        <Col xs={12} sm={6} md={3}>
                            <Form.Group controlId="formStatus">
                                <Form.Label>Status:</Form.Label>
                                <Form.Select name="status" value={filters.status} onChange={handleFilterChange}>
                                    <option value="">Any Status</option>
                                    {[...new Set(cars.map(car => car.Status))].map(status => (
                                        <option key={status} value={status}>{status}</option>
                                    ))}
                                </Form.Select>
                            </Form.Group>
                        </Col>
                    </Row>
                </div>

    
                <div className="sort-options mb-4 d-flex justify-content-left">
                    <Button 
                        variant={sortOrder === 'recent' ? 'secondary' : 'primary'} 
                        onClick={() => handleSort('price')}
                    >
                        Sort by Most Recent
                    </Button>
                    <Button 
                        variant={sortOrder === 'price' ? 'secondary' : 'primary'} 
                        onClick={() => handleSort('recent')} 
                        className="mx-2"
                    >
                        Sort by Price
                    </Button>
                </div>
    
                <Row>
                    {filteredCars.length > 0 ? (
                        filteredCars.map(car => (
                            <Col xs={12} sm={6} md={4} lg={3} key={car.Cid} className="mb-4">
                                <Card>
                                    <Card.Img variant="top" src={car.Img300} />
                                    <Card.Body>
                                        <Card.Title>{car.NameMMT}</Card.Title>
                                        <Card.Text>
                                            <strong>Price:</strong> {car.Prc}<br />
                                            <strong>Year:</strong> {car.Yr}<br />
                                            <strong>Status:</strong> {car.Status}
                                        </Card.Text>
                                        <Button 
                                            variant={highlight.find(c => c.Cid === car.Cid) ? 'danger' : 'primary'} 
                                            onClick={() => highlight.find(c => c.Cid === car.Cid) ? removeHighlight(car) : highlightCar(car)}
                                        >
                                            {highlight.find(c => c.Cid === car.Cid) ? 'Remove Highlight' : 'Highlight'}
                                        </Button>
                                    </Card.Body>
                                </Card>
                            </Col>
                        ))
                    ) : (
                        <Col>
                            <p>{noResultsMessage()}</p>
                        </Col>
                    )}
                </Row>
            </Container>
        </div>
    );
};

export default CarDetailsPage;
