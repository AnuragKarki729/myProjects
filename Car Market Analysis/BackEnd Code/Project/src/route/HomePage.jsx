import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Container, Table, Row, Col,} from 'react-bootstrap';
import { Pie, Bar } from 'react-chartjs-2';
import { Chart as ChartJS, Title, Tooltip, Legend, CategoryScale, LinearScale, BarElement, ArcElement } from 'chart.js';
import 'bootstrap/dist/css/bootstrap.min.css';

ChartJS.register(Title, Tooltip, Legend, CategoryScale, LinearScale, BarElement, ArcElement);

const HomePage = () => {
    const [cars, setCars] = useState([]);
    const [brandModels, setBrandModels] = useState({});
    const [brandCounts, setBrandCounts] = useState({});
    const [modelCounts, setModelCounts] = useState({});
    const [selectedBrand, setSelectedBrand] = useState('');

    const brandKeys = Object.keys(brandModels);
    const halfIndex = Math.ceil(brandKeys.length / 2);

    const firstHalfBrands = brandKeys.slice(0, halfIndex);
    const secondHalfBrands = brandKeys.slice(halfIndex);

    useEffect(() => {
        fetch('./cars.json')
            .then(response => response.json())
            .then(data => {
                const carsData = data.Cars;
                setCars(carsData);

                const brandModels = {};
                const brandCounts = {};

                carsData.forEach(car => {
                    const { MkID, Model, Prc } = car;
                    const brand = data.MMList.find(b => b.mkID === MkID)?.Name || 'Unknown';

                    if (!brandCounts[brand]) {
                        brandCounts[brand] = { count: 0, value: 0 };
                    }
                    brandCounts[brand].count += 1;
                    brandCounts[brand].value += parseInt(Prc.replace(/[^0-9]/g, ''));

                    if (!brandModels[brand]) {
                        brandModels[brand] = {};
                    }
                    if (!brandModels[brand][Model]) {
                        brandModels[brand][Model] = { count: 0, value: 0 };
                    }
                    brandModels[brand][Model].count += 1;
                    brandModels[brand][Model].value += parseInt(Prc.replace(/[^0-9]/g, ''));
                });

                setBrandModels(brandModels);
                setBrandCounts(brandCounts);
                if (selectedBrand) {
                    setModelCounts(brandModels[selectedBrand] || {});
                }
            });
    }, [selectedBrand]);

    const handleBrandSelect = (brand) => {
        setSelectedBrand(brand);
        setModelCounts(brandModels[brand] || {});
    };

    // Data for Pie Chart
    const pieChartData = {
        labels: Object.keys(brandCounts),
        datasets: [{
            data: Object.values(brandCounts).map(b => b.count),
            backgroundColor: [
                '#FF6384',
                '#36A2EB',
                '#FFCE56',
                '#4BC0C0',
                '#FF9F40',
                '#FFCD56',
                '#C9CBCF',
                '#6C757D',
                '#E7E9ED',
                '#F7464A'
            ],
        }],
    };

    // Combine data from both halves of the table
    const combinedModelCounts = {};
    firstHalfBrands.forEach(brand => {
        Object.keys(brandModels[brand] || {}).forEach(model => {
            if (!combinedModelCounts[model]) {
                combinedModelCounts[model] = {};
            }
            combinedModelCounts[model][brand] = brandModels[brand][model]?.count || 0;
        });
    });
    secondHalfBrands.forEach(brand => {
        Object.keys(brandModels[brand] || {}).forEach(model => {
            if (!combinedModelCounts[model]) {
                combinedModelCounts[model] = {};
            }
            combinedModelCounts[model][brand] = brandModels[brand][model]?.count || 0;
        });
    });

    // Generate datasets for Stacked Bar Chart
    const barChartData = {
        labels: brandKeys,
        datasets: Object.keys(combinedModelCounts).map((model, index) => ({
            label: model,
            data: brandKeys.map(brand => combinedModelCounts[model][brand] || 0),
            backgroundColor: `hsl(${index * 30}, 70%, 60%)`, // Different color for each model
        }))
    };

    return (
        <>  
            <Container className="my-4 text-center">
                <h1>Welcome to the Homepage</h1>
                <br />

                {/* Tables for brands and models */}
                <Table className="mx-auto" bordered striped hover>
                    <thead>
                        <tr>
                            <th>Brand</th>
                            <th>Model</th>
                            <th>Number of Cars</th>
                            <th>Value (Baht)</th>
                            <th>Average Price</th>
                        </tr>
                    </thead>
                    <tbody>
                        {firstHalfBrands.map(brand => (
                            <React.Fragment key={brand}>
                                {Object.keys(brandModels[brand]).map(model => (
                                    <tr key={model}>
                                        <td>{model === Object.keys(brandModels[brand])[0] ? brand : ''}</td>
                                        <td>{model}</td>
                                        <td>{brandModels[brand][model]?.count || 0}</td>
                                        <td>{brandModels[brand][model]?.value.toLocaleString() || '0'}</td>
                                        <td>{(brandModels[brand][model]?.value / brandModels[brand][model]?.count).toFixed(2)}</td>
                                    </tr>
                                ))}
                            </React.Fragment>
                        ))}
                    </tbody>
                </Table>
                <Table className="mx-auto" bordered striped hover>
                    <thead>
                        <tr>
                            <th>Brand</th>
                            <th>Model</th>
                            <th>Number of Cars</th>
                            <th>Value (Baht)</th>
                            <th>Average Price</th>
                        </tr>
                    </thead>
                    <tbody>
                        {secondHalfBrands.map(brand => (
                            <React.Fragment key={brand}>
                                {Object.keys(brandModels[brand]).map(model => (
                                    <tr key={model}>
                                        <td>{model === Object.keys(brandModels[brand])[0] ? brand : ''}</td>
                                        <td>{model}</td>
                                        <td>{brandModels[brand][model]?.count || 0}</td>
                                        <td>{brandModels[brand][model]?.value.toLocaleString() || '0'}</td>
                                        <td>{(brandModels[brand][model]?.value / brandModels[brand][model]?.count).toFixed(2)}</td>
                                    </tr>
                                ))}
                            </React.Fragment>
                        ))}
                    </tbody>
                </Table>

                <div className="dashboard-section" style={{ width: '100%', height: '700px', marginBottom:'100px'}}>
                    <h2>Car Distribution by Brand</h2>
                    <Pie data={pieChartData} options={{ responsive: true, maintainAspectRatio: false }}/>
                </div>
                
                <Row className="my-4">
                    <div className="dashboard-section">
                        <h2>Car Distribution by Model</h2>
                        <div style={{ height: '600px', width: '100%' }}> {/* Increase the height */}
                            <Bar
                                data={barChartData}
                                options={{
                                    plugins: {
                                        title: {
                                            display: true,
                                            text: `Cars by Brand and Model`
                                        },
                                        legend: {
                                            display: false, // Hide the legend
                                        },
                                    },
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    scales: {
                                        x: {
                                            stacked: true,
                                        },
                                        y: {
                                            stacked: true,
                                            beginAtZero: true,
                                        },
                                    },
                                }}
                            />
                        </div>
                    </div>
                </Row>
            </Container>
        </>
    );
};

export default HomePage;
